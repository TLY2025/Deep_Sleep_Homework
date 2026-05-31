import argparse
import csv
import json
import os
import random
import sys
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

def _find_transformer_layers(backbone: nn.Module) -> Optional[nn.ModuleList]:
    for name in ["encoder.layer", "model.layers", "layers", "layer", "h", "transformer.h", "backbone.layers"]:
        cur = backbone
        ok = True
        for part in name.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, nn.ModuleList):
            return cur
    for module in backbone.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0 and all(isinstance(x, nn.Module) for x in module):
            return module
    return None


def collect_deep_lora_target_modules(
    backbone: nn.Module,
    last_n_layers: int,
    include_keywords: Optional[List[str]] = None,
) -> List[str]:
    layers = _find_transformer_layers(backbone)
    if layers is None:
        raise ValueError("Cannot find transformer layer stack for LoRA. Please set --lora_target_modules manually.")

    n_layers = len(layers)
    if n_layers <= 0:
        raise ValueError("Found empty transformer layer stack for LoRA.")
    include_keywords = [k.strip() for k in (include_keywords or []) if k.strip()]

    take_n = max(1, min(last_n_layers, n_layers))
    selected_layers = list(layers[-take_n:])

    # Build id->full_name map from backbone namespace for exact PEFT matching.
    name_by_id = {id(m): name for name, m in backbone.named_modules()}
    target_modules: List[str] = []
    for layer in selected_layers:
        for sub_name, sub_module in layer.named_modules():
            if not isinstance(sub_module, nn.Linear):
                continue
            if include_keywords and not any(k in sub_name for k in include_keywords):
                continue
            full_name = name_by_id.get(id(sub_module), "")
            if full_name:
                target_modules.append(full_name)

    if not target_modules:
        raise ValueError(
            "No LoRA target modules found in selected deep layers. "
            "Try reducing --lora_include_keywords constraints."
        )
    return sorted(set(target_modules))


def maybe_wrap_backbone_with_lora(model: "QwenAIDetector", args) -> Optional[Dict[str, object]]:
    if not args.use_lora:
        return None
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for LoRA. Please `pip install peft`.")

    target_modules = args.lora_target_modules
    if not target_modules:
        target_modules = collect_deep_lora_target_modules(
            backbone=model.backbone,
            last_n_layers=args.lora_last_n_layers,
            include_keywords=args.lora_include_keywords,
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    model.backbone = get_peft_model(model.backbone, lora_config)
    print(f"LoRA enabled on {len(target_modules)} target modules (deep last {args.lora_last_n_layers} layers).")
    model.backbone.print_trainable_parameters()
    return {
        "use_lora": True,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_bias": args.lora_bias,
        "lora_last_n_layers": args.lora_last_n_layers,
        "lora_include_keywords": args.lora_include_keywords,
        "lora_target_modules": target_modules,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label(raw: str) -> Optional[int]:
    txt = str(raw).strip()
    if not txt:
        return None
    try:
        value = int(float(txt))
    except ValueError:
        low = txt.lower()
        if low in {"human", "real", "original"}:
            value = 1
        elif low in {"ai", "gpt", "machine", "generated"}:
            value = 0
        else:
            return None
    return value if value in (0, 1) else None


def pick_device(gpu: str) -> torch.device:
    if torch.cuda.is_available():
        if gpu and gpu.strip():
            return torch.device(f"cuda:{int(gpu)}")
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_large_csv_field_limit() -> None:
    max_size = sys.maxsize
    while max_size > 0:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10
    csv.field_size_limit(131072)


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    ensure_large_csv_field_limit()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_rows(rows: List[Dict[str, str]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    ids = list(range(len(rows)))
    random.Random(seed).shuffle(ids)
    val_size = max(1, int(len(rows) * val_ratio))
    val_ids = set(ids[:val_size])
    train_rows = [rows[i] for i in ids if i not in val_ids]
    val_rows = [rows[i] for i in ids if i in val_ids]
    return train_rows, val_rows


def get_question(row: Dict[str, str]) -> str:
    return (row.get("prompt") or row.get("question") or row.get("q") or "").strip()


def get_answer(row: Dict[str, str]) -> str:
    return (row.get("text") or row.get("answer") or row.get("a") or "").strip()


def format_qa_text(question: str, answer: str) -> str:
    return (
        "<TASK:QA>\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
        f"{answer}"
    )


def format_single_text(text: str) -> str:
    return (
        "<TASK:SINGLE>\n"
        "Text:\n"
        f"{text}"
    )


@dataclass
class Sample:
    text: str
    label: int
    task_type: int  # 0: QA, 1: SINGLE


def build_samples(rows: Sequence[Dict[str, str]], single_from_answer_ratio: float) -> List[Sample]:
    out: List[Sample] = []
    for row in tqdm(rows, desc="Building samples"):
        label = parse_label(row.get("label") or row.get("target") or row.get("is_ai") or "")
        if label is None:
            continue
        q = get_question(row)
        a = get_answer(row)
        if q and a:
            out.append(Sample(text=format_qa_text(q, a), label=label, task_type=0))
            if random.random() < max(0.0, min(single_from_answer_ratio, 1.0)):
                out.append(Sample(text=format_single_text(a), label=label, task_type=1))
        elif a:
            out.append(Sample(text=format_single_text(a), label=label, task_type=1))
        elif q:
            out.append(Sample(text=format_single_text(q), label=label, task_type=1))
    return out


class AIDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


class ModernClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        u, v = torch.chunk(self.proj(x), 2, dim=-1) # gate, up
        x = F.silu(u) * v  # SwiGLU
        x = self.drop(x)
        return self.out(x) # down


class QwenAIDetector(nn.Module):
    def __init__(
        self,
        model_dir: str,
        hidden_dim: int = 1024,
        dropout: float = 0.15,
        attn_implementation: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if torch_dtype is None:
            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32

        resolved_attn_impl = attn_implementation
        if attn_implementation == "auto":
            resolved_attn_impl = "eager"
            if torch.cuda.is_available():
                # Prefer FlashAttention-2 when available; otherwise SDPA.
                # (FA2 typically requires fp16/bf16 and Ampere+ GPUs.)
                cc_major = torch.cuda.get_device_capability()[0]
                fa2_ok = (cc_major >= 8) and (torch_dtype in (torch.float16, torch.bfloat16))
                if fa2_ok:
                    try:
                        import flash_attn  # noqa: F401

                        resolved_attn_impl = "flash_attention_2"
                    except Exception:
                        resolved_attn_impl = "sdpa"
                else:
                    resolved_attn_impl = "sdpa"

        self.backbone = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=resolved_attn_impl,
        )
        base_dim = int(self.backbone.config.hidden_size)

        self.attn_score = nn.Linear(base_dim, 1)
        self.pool_fuse = nn.Linear(base_dim * 3, base_dim)
        self.task_embed = nn.Embedding(2, base_dim)
        self.head = ModernClassifierHead(in_dim=base_dim, hidden_dim=hidden_dim, dropout=dropout)

    @staticmethod
    def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1e-6)

    @staticmethod
    def masked_max(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.finfo(hidden.dtype).min
        masked_hidden = hidden.masked_fill(~mask.unsqueeze(-1), neg)
        return masked_hidden.max(dim=1).values

    def attentive_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn_score(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_type: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask_bool = attention_mask.bool()

        mean_pool = self.masked_mean(hidden, attention_mask)
        max_pool = self.masked_max(hidden, mask_bool)
        attn_pool = self.attentive_pool(hidden, mask_bool)
        fused = self.pool_fuse(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))

        fused = fused + self.task_embed(task_type)
        logits = self.head(fused).view(-1)
        result = {"logits": logits}

        if labels is not None:
            # Label convention: 1=human, 0=AI
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            result["loss"] = loss
        return result


def find_transformer_layer_list(module: nn.Module) -> Optional[nn.ModuleList]:
    candidate_paths = [
        "encoder.layer",
        "model.layers",
        "transformer.h",
        "layers",
        "h",
    ]
    for path in candidate_paths:
        cur = module
        valid = True
        for part in path.split("."):
            if not hasattr(cur, part):
                valid = False
                break
            cur = getattr(cur, part)
        if valid and isinstance(cur, nn.ModuleList):
            return cur
    return None


def freeze_backbone_all(model: QwenAIDetector) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone_last_n(model: QwenAIDetector, n: int) -> int:
    layers = find_transformer_layer_list(model.backbone)
    if layers is None or n <= 0:
        return 0
    take = min(len(layers), n)
    for layer in layers[-take:]:
        for p in layer.parameters():
            p.requires_grad = True
    return take


def make_collate(tokenizer, max_len: int):
    def collate_fn(batch: Sequence[Sample]) -> Dict[str, torch.Tensor]:
        enc = tokenizer(
            [x.text for x in batch],
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "task_type": torch.tensor([x.task_type for x in batch], dtype=torch.long),
            "labels": torch.tensor([x.label for x in batch], dtype=torch.float32),
        }
    return collate_fn


def compute_binary_auc(gold: List[int], scores: List[float]) -> float:
    """ROC-AUC for binary labels. Higher score should indicate positive class (gold=1)."""
    n = len(gold)
    if n == 0:
        return float("nan")

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    n_pos = sum(int(g == 1) for g in gold)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = sum(ranks[i] for i in range(n) if gold[i] == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_metrics(
    gold: List[int],
    pred_human: List[int],
    prob_human: Optional[List[float]] = None,
) -> Dict[str, float]:
    total = max(len(gold), 1)
    acc = sum(int(g == p) for g, p in zip(gold, pred_human)) / total

    # Positive class = human (1)
    tp_h = sum(int(g == 1 and p == 1) for g, p in zip(gold, pred_human))
    fp_h = sum(int(g == 0 and p == 1) for g, p in zip(gold, pred_human))
    fn_h = sum(int(g == 1 and p == 0) for g, p in zip(gold, pred_human))
    prec_h = tp_h / (tp_h + fp_h + 1e-12)
    rec_h = tp_h / (tp_h + fn_h + 1e-12)
    f1_h = 2 * prec_h * rec_h / (prec_h + rec_h + 1e-12)

    # AI class metrics (AI = 0 in gold, so invert)
    pred_ai = [1 - p for p in pred_human]
    gold_ai = [1 - g for g in gold]
    tp_ai = sum(int(g == 1 and p == 1) for g, p in zip(gold_ai, pred_ai))
    fp_ai = sum(int(g == 0 and p == 1) for g, p in zip(gold_ai, pred_ai))
    fn_ai = sum(int(g == 1 and p == 0) for g, p in zip(gold_ai, pred_ai))
    prec_ai = tp_ai / (tp_ai + fp_ai + 1e-12)
    rec_ai = tp_ai / (tp_ai + fn_ai + 1e-12)
    f1_ai = 2 * prec_ai * rec_ai / (prec_ai + rec_ai + 1e-12)

    metrics = {
        "accuracy": acc,
        "precision_human": prec_h,
        "recall_human": rec_h,
        "f1_human": f1_h,
        "precision_ai": prec_ai,
        "recall_ai": rec_ai,
        "f1_ai": f1_ai,
    }
    if prob_human is not None:
        prob_ai = [1.0 - p for p in prob_human]
        gold_ai = [1 - g for g in gold]
        metrics["auc_human"] = compute_binary_auc(gold, prob_human)
        metrics["auc_ai"] = compute_binary_auc(gold_ai, prob_ai)
    return metrics


def evaluate(model: QwenAIDetector, dataloader: DataLoader, device: torch.device, human_threshold: float) -> Dict[str, float]:
    model.eval()
    all_gold: List[int] = []
    all_pred_human: List[int] = []
    all_prob_human: List[float] = []
    loss_sum = 0.0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch["labels"].to(device)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "task_type": batch["task_type"].to(device),
                "labels": labels,
            }
            out = model(**inputs)
            probs_human = torch.sigmoid(out["logits"])
            pred_h = (probs_human >= human_threshold).long()
            all_gold.extend(labels.long().cpu().tolist())
            all_pred_human.extend(pred_h.cpu().tolist())
            all_prob_human.extend(probs_human.view(-1).cpu().tolist())
            loss_sum += float(out["loss"].item())
            steps += 1
    metrics = compute_metrics(all_gold, all_pred_human, prob_human=all_prob_human)
    metrics["loss"] = loss_sum / max(steps, 1)
    return metrics


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    auc_human = metrics.get("auc_human", float("nan"))
    auc_ai = metrics.get("auc_ai", float("nan"))
    return (
        f"{prefix}_loss={metrics['loss']:.4f} {prefix}_acc={metrics['accuracy']:.4f} "
        f"{prefix}_f1_ai={metrics['f1_ai']:.4f} {prefix}_f1_human={metrics['f1_human']:.4f} "
        f"{prefix}_auc_human={auc_human:.4f} {prefix}_auc_ai={auc_ai:.4f}"
    )


def build_eval_loaders(args, tokenizer):
    valid_in_loader = None
    valid_ext_loader = None
    valid_in_samples: List[Sample] = []
    valid_ext_samples: List[Sample] = []

    if args.train_csv:
        all_train_rows = read_rows(args.train_csv)
        _, valid_in_rows = split_rows(all_train_rows, args.val_ratio, args.seed)
        valid_in_samples = build_samples(
            valid_in_rows,
            single_from_answer_ratio=1.0 if args.eval_with_single_augment else 0.0,
        )
        if valid_in_samples:
            valid_in_loader = DataLoader(
                AIDataset(valid_in_samples),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=make_collate(tokenizer, args.max_len),
            )

    if args.valid_csv:
        valid_ext_rows = read_rows(args.valid_csv)
        valid_ext_samples = build_samples(
            valid_ext_rows,
            single_from_answer_ratio=1.0 if args.eval_with_single_augment else 0.0,
        )
        if valid_ext_samples:
            valid_ext_loader = DataLoader(
                AIDataset(valid_ext_samples),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=make_collate(tokenizer, args.max_len),
            )

    if valid_in_loader is None and valid_ext_loader is None:
        raise ValueError("Provide --train_csv and/or --valid_csv for evaluation.")
    return valid_in_loader, valid_ext_loader, valid_in_samples, valid_ext_samples


def train_main(args) -> None:
    set_seed(args.seed)
    device = pick_device(args.gpu)

    all_train_rows = read_rows(args.train_csv)
    train_rows, valid_in_rows = split_rows(all_train_rows, args.val_ratio, args.seed)
    valid_ext_rows = read_rows(args.valid_csv) if args.valid_csv else []

    train_samples = build_samples(train_rows, single_from_answer_ratio=args.single_from_answer_ratio)
    valid_in_samples = build_samples(
        valid_in_rows,
        single_from_answer_ratio=1.0 if args.eval_with_single_augment else 0.0,
    )
    valid_ext_samples = build_samples(
        valid_ext_rows,
        single_from_answer_ratio=1.0 if args.eval_with_single_augment else 0.0,
    ) if valid_ext_rows else []

    if not train_samples or not valid_in_samples:
        raise ValueError("No valid samples after preprocessing. Check csv columns and labels.")
    if args.valid_csv and not valid_ext_samples:
        raise ValueError("Provided --valid_csv but got 0 valid samples from it.")

    print(
        f"Samples: train={len(train_samples)}, "
        f"val_in={len(valid_in_samples)}, val_ext={len(valid_ext_samples)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        AIDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_len),
    )
    valid_in_loader = DataLoader(
        AIDataset(valid_in_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, args.max_len),
    )
    valid_ext_loader = None
    if valid_ext_samples:
        valid_ext_loader = DataLoader(
            AIDataset(valid_ext_samples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=make_collate(tokenizer, args.max_len),
        )

    model = QwenAIDetector(
        model_dir=args.base_model,
        hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    )
    lora_meta = None
    if args.use_lora:
        # Keep backbone frozen, only train LoRA adapters + detector head.
        freeze_backbone_all(model)
        lora_meta = maybe_wrap_backbone_with_lora(model, args)
        print("Backbone policy: freeze_all + LoRA adapters")
    else:
        freeze_backbone_all(model)
        unfrozen = unfreeze_backbone_last_n(model, args.unfreeze_last_n)
        print(f"Backbone policy: freeze_all + unfreeze_last_n={unfrozen}")

    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_score = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "best_qwen_ai_detector.pt")

    for epoch in range(1, args.epochs + 1):
        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}_qwen_ai_detector.pt")
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            labels = batch["labels"].to(device)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                task_type=batch["task_type"].to(device),
                labels=labels,
            )
            loss = out["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += float(loss.item())

        train_loss = running_loss / max(len(train_loader), 1)
        metrics_in = evaluate(model, valid_in_loader, device, args.human_threshold)
        metrics_ext = evaluate(model, valid_ext_loader, device, args.human_threshold) if valid_ext_loader is not None else None
        score = metrics_in["f1_ai"]
        if metrics_ext is not None:
            score = 0.5 * (metrics_in["f1_ai"] + metrics_ext["f1_ai"])

        print(
            f"[epoch={epoch}] train_loss={train_loss:.4f} "
            + format_metrics("val_in", metrics_in)
        )
        if metrics_ext is not None:
            print(f"[epoch={epoch}] " + format_metrics("val_ext", metrics_ext))

        if score > best_score:
            best_score = score
            best_ckpt_path = ckpt_path
        torch.save(
            {
                "state_dict": model.state_dict(),
                "base_model": args.base_model,
                "max_len": args.max_len,
                "human_threshold": args.human_threshold,
                "head_hidden_dim": args.head_hidden_dim,
                "dropout": args.dropout,
                "unfreeze_last_n": args.unfreeze_last_n,
                "label_semantics": {"1": "human", "0": "ai"},
                "best_selection": "avg_f1_ai_of_val_in_and_val_ext" if metrics_ext is not None else "f1_ai_of_val_in",
                "best_score": best_score,
                "lora": lora_meta,
            },
            ckpt_path,
        )

    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Best score: {best_score:.4f}")


def eval_main(args) -> None:
    device = pick_device(args.gpu)
    model, tokenizer, ckpt = load_model_for_infer(args.checkpoint, device)
    if args.dtype == "float32":
        model.float()
    elif args.dtype == "float16":
        model.half()
    elif args.dtype == "bfloat16":
        model.bfloat16()
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    model.to(device)
    max_len = args.max_len if args.max_len > 0 else int(ckpt.get("max_len", 512))
    threshold = args.human_threshold if args.human_threshold is not None else float(ckpt.get("human_threshold", 0.5))

    valid_in_loader, valid_ext_loader, valid_in_samples, valid_ext_samples = build_eval_loaders(args, tokenizer)
    print(
        f"Eval checkpoint: {args.checkpoint}\n"
        f"samples: val_in={len(valid_in_samples)}, val_ext={len(valid_ext_samples)}, "
        f"threshold={threshold:.4f}, max_len={max_len}"
    )

    results: Dict[str, Dict[str, float]] = {}
    if valid_in_loader is not None:
        results["val_in"] = evaluate(model, valid_in_loader, device, threshold)
        print(format_metrics("val_in", results["val_in"]))
    if valid_ext_loader is not None:
        results["val_ext"] = evaluate(model, valid_ext_loader, device, threshold)
        print(format_metrics("val_ext", results["val_ext"]))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to: {args.output_json}")


def load_model_for_infer(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    model = QwenAIDetector(
        model_dir=ckpt["base_model"],
        hidden_dim=int(ckpt.get("head_hidden_dim", 1024)),
        dropout=float(ckpt.get("dropout", 0.15)),
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
        model.backbone = get_peft_model(model.backbone, lora_config)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, ckpt


def infer_batch(model, tokenizer, texts: Sequence[str], task_types: Sequence[int], max_len: int, device: torch.device):
    enc = tokenizer(
        list(texts),
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            task_type=torch.tensor(task_types, dtype=torch.long, device=device),
        )
        prob_human = torch.sigmoid(out["logits"]).cpu().tolist()
    prob_ai = [1.0 - p for p in prob_human]
    pred_human = [1 if p >= 0.5 else 0 for p in prob_human]
    pred_ai = [1 - p for p in pred_human]
    return prob_human, prob_ai, pred_human, pred_ai

from typing import List
from tqdm import tqdm
import time
import torch
from openpyxl import Workbook

# 你的原有依赖函数（无需修改）
# pick_device, load_model_for_infer, read_rows, get_question, get_answer
# format_qa_text, format_single_text, infer_batch

def predict_csv_main(args) -> None:
    # 1. 基础配置
    device = pick_device(args.gpu)
    model, tokenizer, ckpt = load_model_for_infer(args.checkpoint, device)
    model.eval()  # 推理模式（必加，提速+省显存）

    if args.dtype == "float32":
        model.float()
    elif args.dtype == "float16":
        model.half()
    elif args.dtype == "bfloat16":
        model.bfloat16()
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    model.to(device)

    max_len = args.max_len if args.max_len > 0 else int(ckpt.get("max_len", 512))
    batch_size = args.batch_size if (hasattr(args, "batch_size") and args.batch_size > 0) else 32
    output_excel = getattr(args, "output_excel", "your-team-name.xlsx")

    # 2. 开始计时
    start_total = time.time()

    # 3. 加载数据（预分配列表，提速）
    rows = read_rows(args.input_csv)
    total_samples = len(rows)
    texts: List[str] = [""] * total_samples
    tasks: List[int] = [0] * total_samples

    for idx, row in enumerate(tqdm(rows, desc="Loading Data")):
        q = get_question(row)
        a = get_answer(row)
        if q and a and not args.force_single:
            texts[idx] = format_qa_text(q, a)
            tasks[idx] = 0
        else:
            txt = a or q
            texts[idx] = format_single_text(txt)
            tasks[idx] = 1

    # 4. 分批推理（核心：无梯度计算，极致提速）
    all_prob_human: List[float] = [0.0] * total_samples
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size), desc="Batch Predicting"):
            b_slice = slice(i, i + batch_size)
            prob_human_batch, _, _, _ = infer_batch(
                model, tokenizer, texts[b_slice], tasks[b_slice], max_len, device
            )
            all_prob_human[b_slice] = prob_human_batch

    # 5. 计算总耗时
    total_time = round(time.time() - start_total, 2)

    # ===================== 极简稳定Excel写入（零报错！） =====================
    wb = Workbook()

    # Sheet1: predictions （严格按要求格式）
    ws1 = wb.active
    ws1.title = "predictions"
    ws1.append(["prompt", "text_prediction"])
    for prompt, prob in zip(texts, all_prob_human):
        ws1.append([prompt, round(prob, 6)])

    # Sheet2: time （严格按要求格式）
    ws2 = wb.create_sheet(title="time")
    ws2.append(["Data Volume", "Time"])
    ws2.append([total_samples, total_time])

    # 保存（极速写入，无任何样式开销）
    wb.save(output_excel)

    # ===================== 结束 =====================
    print(f"\n✅ 全部完成！结果已保存：{output_excel}")
    print(f"📊 总样本数：{total_samples}")
    print(f"⏱️ 总耗时：{total_time} 秒")


def predict_text_main(args) -> None:
    device = pick_device(args.gpu)
    model, tokenizer, ckpt = load_model_for_infer(args.checkpoint, device)
    max_len = args.max_len if args.max_len > 0 else int(ckpt.get("max_len", 512))
    threshold = args.human_threshold if args.human_threshold is not None else float(ckpt.get("human_threshold", 0.5))

    if args.question and args.answer:
        text = format_qa_text(args.question, args.answer)
        task = 0
    else:
        if not args.text:
            raise ValueError("Provide --text or both --question and --answer.")
        text = format_single_text(args.text)
        task = 1

    prob_human, prob_ai, _, _ = infer_batch(model, tokenizer, [text], [task], max_len, device)
    ph = prob_human[0]
    pa = prob_ai[0]
    pred_h = 1 if ph >= threshold else 0
    pred_a = 1 - pred_h

    print(f"prob_human={ph:.6f}")
    print(f"prob_ai={pa:.6f}")
    print(f"pred_human={pred_h}")
    print(f"pred_ai={pred_a}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Qwen3-0.6B-Embedding AI text detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--base_model", type=str, default="/mnt/jfzn/zjh/playground/Qwen3-0.6B-Embedding")
    p_train.add_argument("--train_csv", type=str, required=True)
    p_train.add_argument("--valid_csv", type=str, default="", help="Extra external validation csv (optional).")
    p_train.add_argument("--val_ratio", type=float, default=0.1, help="Split ratio taken from train_csv as internal validation set.")
    p_train.add_argument("--output_dir", type=str, required=True)
    p_train.add_argument("--single_from_answer_ratio", type=float, default=0.35)
    p_train.add_argument("--eval_with_single_augment", action=argparse.BooleanOptionalAction, default=False)
    p_train.add_argument("--max_len", type=int, default=512)
    p_train.add_argument("--batch_size", type=int, default=12)
    p_train.add_argument("--epochs", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight_decay", type=float, default=0.01)
    p_train.add_argument("--max_grad_norm", type=float, default=1.0)
    p_train.add_argument("--dropout", type=float, default=0.15)
    p_train.add_argument("--head_hidden_dim", type=int, default=1024)
    p_train.add_argument("--human_threshold", type=float, default=0.5)
    p_train.add_argument("--unfreeze_last_n", type=int, default=2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--gpu", type=str, default="")

    p_pred_csv = sub.add_parser("predict_csv")
    p_pred_csv.add_argument("--checkpoint", type=str, required=True)
    p_pred_csv.add_argument("--input_csv", type=str, required=True)
    p_pred_csv.add_argument("--output_csv", type=str, required=True)
    p_pred_csv.add_argument("--max_len", type=int, default=0)
    p_pred_csv.add_argument("--human_threshold", type=float, default=None)
    p_pred_csv.add_argument("--force_single", action=argparse.BooleanOptionalAction, default=False)
    p_pred_csv.add_argument("--gpu", type=str, default="")
    p_pred_csv.add_argument("--batch_size", type=int, default=32)
    p_pred_csv.add_argument("--dtype", type=str, default="float32")

    p_pred_text = sub.add_parser("predict_text")
    p_pred_text.add_argument("--checkpoint", type=str, required=True)
    p_pred_text.add_argument("--text", type=str, default="")
    p_pred_text.add_argument("--question", type=str, default="")
    p_pred_text.add_argument("--answer", type=str, default="")
    p_pred_text.add_argument("--max_len", type=int, default=0)
    p_pred_text.add_argument("--human_threshold", type=float, default=None)
    p_pred_text.add_argument("--gpu", type=str, default="")

    p_eval = sub.add_parser("eval", help="Evaluate a pretrained checkpoint on validation csv(s) with AUC.")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--train_csv", type=str, default="", help="Train csv; split val_ratio as internal validation.")
    p_eval.add_argument("--valid_csv", type=str, default="", help="External validation csv.")
    p_eval.add_argument("--val_ratio", type=float, default=0.1)
    p_eval.add_argument("--eval_with_single_augment", action=argparse.BooleanOptionalAction, default=False)
    p_eval.add_argument("--max_len", type=int, default=0)
    p_eval.add_argument("--batch_size", type=int, default=32)
    p_eval.add_argument("--human_threshold", type=float, default=None)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--gpu", type=str, default="")
    p_eval.add_argument("--output_json", type=str, default="")
    p_eval.add_argument("--dtype", type=str, default="float32")

    p_train.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=False)
    p_train.add_argument("--lora_r", type=int, default=8)
    p_train.add_argument("--lora_alpha", type=int, default=16)
    p_train.add_argument("--lora_dropout", type=float, default=0.05)
    p_train.add_argument("--lora_bias", type=str, choices=["none", "all", "lora_only"], default="none")
    p_train.add_argument("--lora_last_n_layers", type=int, default=4)
    p_train.add_argument(
        "--lora_include_keywords",
        nargs="+",
        default=["query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "dense"],
        help="Only apply LoRA to module names containing these keywords in deep layers.",
    )
    p_train.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=None,
        help="Explicit LoRA target modules. If set, overrides auto deep-layer selection.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "eval":
        eval_main(args)
    elif args.cmd == "predict_csv":
        predict_csv_main(args)
    elif args.cmd == "predict_text":
        predict_text_main(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=4 nohup python modeling/train_qwen_ai_detector.py train \
    --base_model /mnt/jfzn/zjh/playground/Qwen3-0.6B-Embedding \
    --train_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/datasets/merged_dataset.csv \
    --valid_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/datasets/val/UCAS_AISAD_TEXT-val.csv \
    --output_dir /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/checkpoints/qwen_ai_detector \
    --val_ratio 0.05 \
    --single_from_answer_ratio 0.35 \
    --eval_with_single_augment \
    --max_len 2048 \
    --batch_size 32 \
    --epochs 2 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dropout 0.15 \
    --head_hidden_dim 1024 \
    --human_threshold 0.5 \
    --unfreeze_last_n 2 \
    --seed 42  > /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/logs/qwen_ai_detector.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python modeling/train_qwen_ai_detector.py train \
    --base_model /mnt/jfzn/zjh/playground/Qwen3-0.6B-Embedding \
    --train_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/datasets/merged_dataset.csv \
    --valid_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/datasets/val/UCAS_AISAD_TEXT-val.csv \
    --output_dir /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/checkpoints/qwen_ai_detector_lora \
    --val_ratio 0.05 \
    --single_from_answer_ratio 0.35 \
    --eval_with_single_augment \
    --max_len 2048 \
    --batch_size 32 \
    --epochs 2 \
    --use_lora \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
    --lora_last_n_layers 12 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dropout 0.15 \
    --head_hidden_dim 1024 \
    --human_threshold 0.5 \
    --seed 42  > /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/logs/qwen_ai_detector_lora.log 2>&1 &

nohup python modeling/train_qwen_ai_detector.py predict_csv \
    --checkpoint /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/checkpoints/qwen_ai_detector_lora/epoch_1_qwen_ai_detector.pt \
    --input_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/datasets/test1/UCAS_AISAD_TEXT-test1.csv \
    --output_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/outputs/qwen_ai_detector_lora_predict.csv \
    --max_len 2048 \
    --batch_size 248 \
    --human_threshold 0.5 \
    --dtype bfloat16 \
    --gpu 5  > /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/logs/qwen_ai_detector_lora_predict.log 2>&1 &

nohup python modeling/train_qwen_ai_detector.py eval \
  --checkpoint /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/checkpoints/qwen_ai_detector_lora/epoch_1_qwen_ai_detector.pt \
  --valid_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/datasets/val/UCAS_AISAD_TEXT-val.csv \
  --max_len 2048 \
  --batch_size 32 \
  --gpu 6 \
  --dtype bfloat16 \
  --output_json /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/logs/qwen_ai_detector_eval_metrics.json > /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/logs/qwen_ai_detector_lora_predict_auc.log 2>&1 &
"""