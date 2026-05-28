import argparse
import csv
import os
import pickle
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from test_ai_text_detector import (
    DesklibHybridAIDetectionModel,
    LoraConfig,
    FeatureScaler,
    TextDataset,
    TextFeatureEngineer,
    build_examples,
    build_input_text,
    compute_metrics,
    get_peft_model,
    load_hybrid_checkpoint,
    load_csv_rows,
    make_collate_fn,
    normalize_text_modes,
    parse_label,
    parse_gpu_ids,
    pick_device,
)


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        h = outputs[0]
        mask = attention_mask.unsqueeze(-1).expand(h.size()).float()
        pooled = torch.sum(h * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        logits = self.classifier(pooled)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())
        return out


def _load_hybrid_checkpoint_compat(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = AutoConfig.from_pretrained(ckpt["model_dir"])
    model = DesklibHybridAIDetectionModel(
        config,
        extra_feature_dim=int(ckpt.get("extra_feature_dim", 17)),
        mlp_hidden_dim=int(ckpt.get("mlp_hidden_dim", 256)),
        dropout=float(ckpt.get("dropout", 0.1)),
        fusion_alpha_init=float(ckpt.get("fusion_alpha_init", -1.5)),
    )
    lora_meta = ckpt.get("lora") or {}
    if lora_meta.get("use_lora", False):
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("Checkpoint uses LoRA but peft is not installed. Please `pip install peft`.")
        lora_config = LoraConfig(
            r=int(lora_meta.get("lora_r", 8)),
            lora_alpha=int(lora_meta.get("lora_alpha", 16)),
            target_modules=lora_meta.get("lora_target_modules", []),
            lora_dropout=float(lora_meta.get("lora_dropout", 0.05)),
            bias=str(lora_meta.get("lora_bias", "none")),
        )
        model.model = get_peft_model(model.model, lora_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt["model_dir"])
    scaler = FeatureScaler.from_state_dict(ckpt["scaler"])
    return model, tokenizer, scaler, ckpt


def load_model_or_checkpoint(model_or_checkpoint: str, device: torch.device):
    if os.path.isdir(model_or_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_or_checkpoint)
        model = DesklibAIDetectionModel.from_pretrained(model_or_checkpoint)
        model.to(device)
        model.eval()
        meta = {
            "kind": "base_model_dir",
            "model_dir": model_or_checkpoint,
            "train_text_modes": ["qa_concat"],
            "threshold": 0.5,
            "max_len": 768,
            "use_perplexity": False,
            "ppl_model_dir": "",
            "ppl_max_len": 512,
        }
        return model, tokenizer, None, meta

    try:
        model, tokenizer, scaler, ckpt = load_hybrid_checkpoint(model_or_checkpoint, device)
        ckpt["kind"] = "hybrid_checkpoint"
        return model, tokenizer, scaler, ckpt
    except pickle.UnpicklingError:
        model, tokenizer, scaler, ckpt = _load_hybrid_checkpoint_compat(model_or_checkpoint, device)
        ckpt["kind"] = "hybrid_checkpoint"
        return model, tokenizer, scaler, ckpt


class TextOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def make_text_only_collate_fn(tokenizer, max_len: int):
    def collate_fn(batch: List[str]):
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    return collate_fn


def evaluate_checkpoint_on_csv(
    model_or_checkpoint: str,
    csv_path: str,
    text_mode: str,
    batch_size: int,
    max_len: int,
    threshold: Optional[float],
    gpu_ids: str,
    output_csv: str,
    reverse_label: bool,
    reverse_output: bool,
) -> None:
    device = pick_device(parse_gpu_ids(gpu_ids))
    model, tokenizer, scaler, ckpt = load_model_or_checkpoint(model_or_checkpoint, device)

    mode = text_mode.strip()
    if not mode:
        mode = normalize_text_modes(ckpt.get("train_text_modes", ["qa_concat"]))[0]

    used_threshold = float(ckpt.get("threshold", 0.5)) if threshold is None else threshold
    used_max_len = int(ckpt.get("max_len", 768)) if max_len <= 0 else max_len

    rows = load_csv_rows(csv_path)
    if ckpt.get("kind") == "hybrid_checkpoint":
        feature_engineer = TextFeatureEngineer(
            use_perplexity=bool(ckpt.get("use_perplexity", False)),
            ppl_model_dir=ckpt.get("ppl_model_dir", ""),
            ppl_max_len=int(ckpt.get("ppl_max_len", 512)),
            device=device,
        )
        examples = build_examples(rows, [mode], feature_engineer, require_label=False)
        for ex in examples:
            ex.features = scaler.transform(ex.features)
        dataloader = DataLoader(
            TextDataset(examples),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(tokenizer, used_max_len),
        )
    else:
        texts = [build_input_text(row, mode) for row in rows]
        dataloader = DataLoader(
            TextOnlyDataset(texts),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=make_text_only_collate_fn(tokenizer, used_max_len),
        )

    preds: List[Tuple[float, int]] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)["logits"]
            probs = torch.sigmoid(logits).view(-1).cpu().tolist()
            preds.extend((float(p), int(p >= used_threshold)) for p in probs)
    if reverse_output: # 1 stands for human
        preds = [(1 - p, 1 - y_hat) for p, y_hat in preds]
    gold: List[int] = []
    pred: List[int] = []
    for row, (_, y_hat) in zip(rows, preds):
        y = parse_label(row)
        if y is None:
            continue
        if reverse_label:
            y = 1 - y
        gold.append(y)
        pred.append(y_hat)

    print(f"rows={len(rows)}, predicted={len(preds)}, mode={mode}, threshold={used_threshold:.4f}")
    if gold:
        metrics = compute_metrics(gold, pred)
        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision_ai={metrics['precision_ai']:.4f}, "
            f"recall_ai={metrics['recall_ai']:.4f}, "
            f"f1_ai={metrics['f1_ai']:.4f}"
        )
    else:
        print("No valid label column found; metrics skipped.")

    if output_csv:
        fieldnames = list(rows[0].keys()) + ["prob_ai", "pred_label"]
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, (prob, label) in zip(rows, preds):
                out = dict(row)
                out["prob_ai"] = f"{prob:.6f}"
                out["pred_label"] = str(label)
                writer.writerow(out)
        print(f"Saved predictions to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate hybrid checkpoint or base model dir on labeled CSV")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_or_checkpoint", type=str, default="")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--text_mode", type=str, choices=["qa_concat", "a_only"], default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--reverse_label", action="store_true")
    parser.add_argument("--reverse_output", action="store_true")
    args = parser.parse_args()
    source = args.model_or_checkpoint or args.checkpoint
    if not source:
        raise ValueError("Provide --model_or_checkpoint (or legacy --checkpoint).")

    evaluate_checkpoint_on_csv(
        model_or_checkpoint=source,
        csv_path=args.csv_path,
        text_mode=args.text_mode,
        batch_size=args.batch_size,
        max_len=args.max_len,
        threshold=args.threshold,
        gpu_ids=args.gpu_ids,
        output_csv=args.output_csv,
        reverse_label=args.reverse_label,
        reverse_output=args.reverse_output
    )


if __name__ == "__main__":
    main()
