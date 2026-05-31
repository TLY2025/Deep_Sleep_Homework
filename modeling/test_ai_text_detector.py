import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label(row: Dict[str, str]) -> Optional[int]:
    raw = (row.get("label") or row.get("target") or row.get("is_ai") or "").strip()
    if not raw:
        return None
    try:
        v = int(float(raw))
    except ValueError:
        low = raw.lower()
        if low in {"ai", "generated", "machine", "gpt"}:
            v = 1
        elif low in {"human", "real", "original"}:
            v = 0
        else:
            return None
    return v if v in (0, 1) else None


def build_input_text(row: Dict[str, str], text_mode: str) -> str:
    prompt = (row.get("prompt") or row.get("question") or "").strip()
    answer = (row.get("text") or row.get("answer") or row.get("a") or "").strip()
    if text_mode == "a_only":
        return answer
    if prompt and answer:
        return f"{prompt}</s>{answer}"
    return answer or prompt


def normalize_text_modes(modes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for mode in modes:
        if mode not in {"qa_concat", "a_only"}:
            raise ValueError(f"Unsupported text mode: {mode}")
        if mode not in out:
            out.append(mode)
    if not out:
        out = ["qa_concat"]
    return out


def merge_path_args(single_path: str, path_list: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    if single_path:
        out.append(single_path)
    if path_list:
        out.extend(path_list)
    return out


def load_csv_rows(csv_path: str, limit_rows: int = 0) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit_rows and limit_rows > 0:
        return rows[:limit_rows]
    return rows


def load_multi_csv_rows(paths: List[str], limit_rows_per_csv: int = 0) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for p in paths:
        merged.extend(load_csv_rows(p, limit_rows=limit_rows_per_csv))
    return merged


def parse_gpu_ids(gpu_ids_raw: str) -> List[int]:
    if not gpu_ids_raw.strip():
        return []
    return [int(x.strip()) for x in gpu_ids_raw.split(",") if x.strip()]


def pick_device(gpu_ids: List[int]) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_ids:
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cuda")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


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

    n = len(layers)
    if n <= 0:
        raise ValueError("Found empty transformer layer stack for LoRA.")
    take_n = max(1, min(last_n_layers, n))
    start = n - take_n
    include_keywords = [k.strip() for k in (include_keywords or []) if k.strip()]

    target_modules: List[str] = []
    for idx in range(start, n):
        layer = layers[idx]
        for sub_name, sub_module in layer.named_modules():
            if not isinstance(sub_module, nn.Linear):
                continue
            if include_keywords and not any(k in sub_name for k in include_keywords):
                continue
            full_name = f"{idx}.{sub_name}" if sub_name else str(idx)
            target_modules.append(full_name)

    if not target_modules:
        raise ValueError(
            "No LoRA target modules found in selected deep layers. "
            "Try reducing --lora_include_keywords constraints."
        )
    return sorted(set(target_modules))


def maybe_wrap_backbone_with_lora(model: nn.Module, args) -> Optional[Dict[str, object]]:
    if not args.use_lora:
        return None
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for LoRA. Please `pip install peft`.")

    core_model = unwrap_model(model)
    backbone = core_model.model
    target_modules = args.lora_target_modules
    if not target_modules:
        target_modules = collect_deep_lora_target_modules(
            backbone=backbone,
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
    core_model.model = get_peft_model(backbone, lora_config)
    print(f"LoRA enabled on {len(target_modules)} target modules (deep last {args.lora_last_n_layers} layers).")
    core_model.model.print_trainable_parameters()
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


class FeatureDiskCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_cache (
                cache_key TEXT PRIMARY KEY,
                features_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[List[float]]:
        cur = self.conn.execute("SELECT features_json FROM feature_cache WHERE cache_key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, features: List[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO feature_cache(cache_key, features_json, created_at) VALUES (?, ?, strftime('%s','now'))",
            (key, json.dumps(features, ensure_ascii=False)),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def build_feature_cache_key(
    text: str,
    text_mode: str,
    feature_version: str,
    use_perplexity: bool,
    ppl_model_dir: str,
    ppl_max_len: int,
) -> str:
    payload = {
        "feature_version": feature_version,
        "text_mode": text_mode,
        "text": text,
        "use_perplexity": use_perplexity,
        "ppl_model_dir": ppl_model_dir if use_perplexity else "",
        "ppl_max_len": ppl_max_len if use_perplexity else 0,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def try_get_features_from_row(
    row: Dict[str, str],
    feature_names: List[str],
    text_mode: str,
) -> Optional[List[float]]:
    # First try mode-specific columns: feat_<mode>_<name>
    mode_cols = [f"feat_{text_mode}_{name}" for name in feature_names]
    if all(c in row and str(row.get(c, "")).strip() != "" for c in mode_cols):
        try:
            return [float(row[c]) for c in mode_cols]
        except ValueError:
            pass

    # Fallback to generic columns: feat_<name>
    generic_cols = [f"feat_{name}" for name in feature_names]
    if all(c in row and str(row.get(c, "")).strip() != "" for c in generic_cols):
        try:
            return [float(row[c]) for c in generic_cols]
        except ValueError:
            pass

    return None


class TextFeatureEngineer:
    def __init__(
        self,
        use_perplexity: bool = False,
        ppl_model_dir: str = "",
        ppl_max_len: int = 512,
        device: Optional[torch.device] = None,
    ):
        self.use_perplexity = use_perplexity and bool(ppl_model_dir)
        self.ppl_model_dir = ppl_model_dir
        self.ppl_max_len = ppl_max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ppl_tokenizer = None
        self.ppl_model = None

        if self.use_perplexity:
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_dir)
            self.ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_dir)
            self.ppl_model.to(self.device)
            self.ppl_model.eval()

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "char_count", # char 数量
            "word_count", # 单词数量
            "avg_word_len", # 平均单词长度
            "sentence_count",
            "avg_sent_len", # 平均句子长度
            "punct_ratio", # 标点符号比例
            "upper_ratio", # 大写字母比例
            "digit_ratio", # 数字比例
            "unique_word_ratio",
            "sent_len_std", # 句子长度标准差
            "sent_len_cv", # 句子长度变异系数
            "word_len_std",
            "repeated_bigram_ratio", # 重复二元组比例
            "char_entropy", # 字符熵
            "word_entropy", # 单词熵
            "perplexity", # 困惑度
            "log_perplexity", # 困惑度对数
        ]

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"[.!?。！？\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))

    @staticmethod
    def _entropy(items: List[str]) -> float:
        if not items:
            return 0.0
        freq: Dict[str, int] = {}
        for item in items:
            freq[item] = freq.get(item, 0) + 1
        total = len(items)
        ent = 0.0
        for c in freq.values():
            p = c / total
            ent -= p * math.log(p + 1e-12)
        return ent

    def _compute_perplexity(self, text: str) -> float:
        if not self.use_perplexity or self.ppl_tokenizer is None or self.ppl_model is None:
            return 0.0
        with torch.no_grad():
            enc = self.ppl_tokenizer(
                text,
                truncation=True,
                max_length=self.ppl_max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.ppl_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = float(out.loss.detach().cpu().item())
        return math.exp(min(20.0, loss))

    def transform_one(self, text: str) -> List[float]:
        words = self._tokenize_words(text)
        sents = self._split_sentences(text)
        chars = list(text)
        char_count = len(text)
        word_count = len(words)
        sent_count = len(sents)

        avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0.0
        sent_lens = [len(self._tokenize_words(s)) for s in sents]
        avg_sent_len = sum(sent_lens) / sent_count if sent_count else 0.0
        punct_count = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))
        upper_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        unique_word_ratio = float(len(set(words)) / word_count) if word_count else 0.0

        sent_len_std = self._std([float(x) for x in sent_lens]) if sent_lens else 0.0
        sent_len_cv = sent_len_std / (avg_sent_len + 1e-12)
        word_len_std = self._std([float(len(w)) for w in words]) if words else 0.0
        if len(words) >= 2:
            bigrams = [f"{words[i]}__{words[i + 1]}" for i in range(len(words) - 1)]
            repeated_bigram_ratio = 1.0 - len(set(bigrams)) / max(len(bigrams), 1)
        else:
            repeated_bigram_ratio = 0.0

        ppl = self._compute_perplexity(text)
        log_ppl = math.log(ppl + 1e-12) if ppl > 0 else 0.0

        return [
            float(char_count),
            float(word_count),
            float(avg_word_len),
            float(sent_count),
            float(avg_sent_len),
            float(punct_count / max(char_count, 1)),
            float(upper_count / max(char_count, 1)),
            float(digit_count / max(char_count, 1)),
            unique_word_ratio,
            float(sent_len_std),
            float(sent_len_cv),
            float(word_len_std),
            float(repeated_bigram_ratio),
            float(self._entropy(chars)),
            float(self._entropy(words)),
            float(ppl),
            float(log_ppl),
        ]


class FeatureScaler:
    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        self.mean = mean
        self.std = std

    def fit(self, features: List[List[float]]) -> None:
        x = torch.tensor(features, dtype=torch.float32)
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.mean = mean.tolist()
        self.std = std.tolist()

    def transform(self, x: List[float]) -> List[float]:
        if self.mean is None or self.std is None:
            raise ValueError("FeatureScaler not fitted")
        return [(v - m) / (s + 1e-12) for v, m, s in zip(x, self.mean, self.std)]

    def state_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean or [], "std": self.std or []}

    @classmethod
    def from_state_dict(cls, state: Dict[str, List[float]]) -> "FeatureScaler":
        return cls(mean=state.get("mean"), std=state.get("std"))


class GatedMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)

        x = gate * up
        x = self.dropout(x)

        x = self.down_proj(x)

        return x

class DesklibHybridAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config,
        extra_feature_dim: int = 17,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
        fusion_alpha_init: float = -1.5,
    ):
        super().__init__(config)
        self.model = AutoModel.from_config(config)

        # Original detector head (kept to inherit pretrained classifier weights).
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Modern fusion head for [pooled_text ; handcrafted_features].
        fused_dim = config.hidden_size + extra_feature_dim
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_up = nn.Linear(fused_dim, mlp_hidden_dim * 2)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_out = nn.Linear(mlp_hidden_dim, 1)

        # Learnable blend between old head and modern fused head.
        self.fusion_alpha = nn.Parameter(torch.tensor(float(fusion_alpha_init)))
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        extra_features=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        h = outputs[0]
        mask = attention_mask.unsqueeze(-1).expand(h.size()).float()
        pooled = torch.sum(h * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

        old_logits = self.classifier(pooled)

        if extra_features is None:
            extra_features = torch.zeros(pooled.shape[0], 17, dtype=pooled.dtype, device=pooled.device)
        fused = torch.cat([pooled, extra_features.to(pooled.dtype)], dim=-1)

        fused = self.fusion_norm(fused)
        uv = self.fusion_up(self.fusion_dropout(fused))
        u, v = torch.chunk(uv, chunks=2, dim=-1)
        hidden = F.silu(u) * v
        hidden = self.fusion_dropout(hidden)
        new_logits = self.fusion_out(hidden)

        alpha = torch.sigmoid(self.fusion_alpha)
        logits = (1.0 - alpha) * old_logits + alpha * new_logits
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())
        return out


@dataclass
class Example:
    text: str
    label: Optional[int]
    features: List[float]


class TextDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def build_examples(
    rows: List[Dict[str, str]],
    text_modes: List[str],
    feature_engineer: TextFeatureEngineer,
    require_label: bool,
    feature_cache: Optional[FeatureDiskCache] = None,
    feature_version: str = "v1",
) -> List[Example]:
    examples: List[Example] = []
    feature_names = feature_engineer.feature_names()
    cache_hit = 0
    cache_miss = 0
    row_hit = 0
    computed = 0
    for row in tqdm(rows, desc="Extract features"):
        label = parse_label(row) if require_label else None
        if require_label and label is None:
            continue
        for mode in text_modes:
            text = build_input_text(row, mode)
            if not text:
                continue

            features = try_get_features_from_row(row, feature_names, mode)
            if features is not None:
                row_hit += 1
            else:
                features = None
                if feature_cache is not None:
                    key = build_feature_cache_key(
                        text=text,
                        text_mode=mode,
                        feature_version=feature_version,
                        use_perplexity=feature_engineer.use_perplexity,
                        ppl_model_dir=feature_engineer.ppl_model_dir,
                        ppl_max_len=feature_engineer.ppl_max_len,
                    )
                    features = feature_cache.get(key)
                    if features is not None:
                        cache_hit += 1
                    else:
                        cache_miss += 1

                if features is None:
                    features = feature_engineer.transform_one(text)
                    computed += 1
                    if feature_cache is not None:
                        feature_cache.set(key, features)

            examples.append(Example(text=text, label=label, features=features))

    if feature_cache is not None:
        feature_cache.commit()
    print(
        f"Feature reuse: row_hit={row_hit}, cache_hit={cache_hit}, "
        f"cache_miss={cache_miss}, computed={computed}"
    )
    return examples


def split_examples(examples: List[Example], val_ratio: float, seed: int) -> Tuple[List[Example], List[Example]]:
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    val_size = max(1, int(len(examples) * val_ratio))
    val_idx = set(idx[:val_size])
    train_ex = [examples[i] for i in idx if i not in val_idx]
    val_ex = [examples[i] for i in idx if i in val_idx]
    return train_ex, val_ex


def make_collate_fn(tokenizer, max_len: int):
    def collate_fn(batch: List[Example]) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            [x.text for x in batch],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded["extra_features"] = torch.tensor([x.features for x in batch], dtype=torch.float32)
        if all(x.label is not None for x in batch):
            encoded["labels"] = torch.tensor([x.label for x in batch], dtype=torch.float32)
        return encoded

    return collate_fn


def compute_metrics(gold: List[int], pred: List[int]) -> Dict[str, float]:
    correct = sum(int(g == p) for g, p in zip(gold, pred))
    tp = sum(int(g == 1 and p == 1) for g, p in zip(gold, pred))
    fp = sum(int(g == 0 and p == 1) for g, p in zip(gold, pred))
    fn = sum(int(g == 1 and p == 0) for g, p in zip(gold, pred))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"accuracy": correct / max(len(gold), 1), "precision_ai": precision, "recall_ai": recall, "f1_ai": f1}


def evaluate(model, dataloader, device, threshold: float) -> Dict[str, float]:
    model.eval()
    gold: List[int] = []
    pred: List[int] = []
    loss_sum = 0.0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate"):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, labels=labels)
            probs = torch.sigmoid(out["logits"]).view(-1)
            preds = (probs >= threshold).long()
            gold.extend(labels.long().cpu().tolist())
            pred.extend(preds.cpu().tolist())
            loss = out["loss"]
            if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                loss = loss.mean()
            loss_sum += float(loss.item())
            steps += 1
    m = compute_metrics(gold, pred)
    m["loss"] = loss_sum / max(steps, 1)
    return m


def prepare_train_val(args, feature_engineer: TextFeatureEngineer):
    train_modes = normalize_text_modes(args.train_text_modes)
    val_modes = normalize_text_modes(args.val_text_modes or train_modes)

    train_csvs = merge_path_args(args.train_csv, args.train_csvs)
    val_csvs = merge_path_args(args.val_csv, args.val_csvs)
    all_csvs = merge_path_args(args.csv_path, args.csv_paths)

    feature_cache = FeatureDiskCache(args.feature_cache_db) if args.use_feature_cache else None

    if train_csvs and val_csvs:
        train_rows = load_multi_csv_rows(train_csvs, limit_rows_per_csv=args.smoke_rows_per_csv)
        val_rows = load_multi_csv_rows(val_csvs, limit_rows_per_csv=args.smoke_rows_per_csv)
        train_examples = build_examples(
            train_rows,
            train_modes,
            feature_engineer,
            True,
            feature_cache=feature_cache,
            feature_version=args.feature_cache_version,
        )
        val_examples = build_examples(
            val_rows,
            val_modes,
            feature_engineer,
            True,
            feature_cache=feature_cache,
            feature_version=args.feature_cache_version,
        )
    elif all_csvs:
        rows = load_multi_csv_rows(all_csvs, limit_rows_per_csv=args.smoke_rows_per_csv)
        examples = build_examples(
            rows,
            train_modes,
            feature_engineer,
            True,
            feature_cache=feature_cache,
            feature_version=args.feature_cache_version,
        )
        train_examples, val_examples = split_examples(examples, args.val_ratio, args.seed)
    else:
        raise ValueError("Provide --csv_path/--csv_paths or both train/val csvs.")

    if feature_cache is not None:
        feature_cache.close()

    if not train_examples or not val_examples:
        raise ValueError("Empty train/val examples. Check csv and labels.")
    return train_examples, val_examples, train_modes


def train(args) -> None:
    set_seed(args.seed)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = pick_device(gpu_ids)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    feature_engineer = TextFeatureEngineer(
        use_perplexity=args.use_perplexity,
        ppl_model_dir=args.ppl_model_dir,
        ppl_max_len=args.ppl_max_len,
        device=device,
    )

    train_examples, val_examples, train_modes = prepare_train_val(args, feature_engineer)

    scaler = FeatureScaler()
    scaler.fit([x.features for x in train_examples])
    for e in train_examples:
        e.features = scaler.transform(e.features)
    for e in val_examples:
        e.features = scaler.transform(e.features)

    train_loader = DataLoader(
        TextDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, args.max_len),
    )
    val_loader = DataLoader(
        TextDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, args.max_len),
    )

    model = DesklibHybridAIDetectionModel.from_pretrained(
        args.model_dir,
        extra_feature_dim=len(train_examples[0].features),
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
        fusion_alpha_init=args.fusion_alpha_init,
        ignore_mismatched_sizes=True,
    )
    lora_meta = maybe_wrap_backbone_with_lora(model, args)

    if torch.cuda.is_available() and args.data_parallel and len(gpu_ids) > 1:
        model.to(device)
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        model.to(device)

    core_model = unwrap_model(model)
    if args.freeze_backbone and not args.use_lora:
        for p in core_model.model.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    best_ckpt = os.path.join(args.output_dir, "best_hybrid_detector.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, labels=labels)
            loss = out["loss"]
            if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            run_loss += float(loss.item())

        val_metrics = evaluate(model, val_loader, device, args.threshold)
        train_loss = run_loss / max(len(train_loader), 1)
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_ai={val_metrics['f1_ai']:.4f}"
        )

        if val_metrics["f1_ai"] > best_f1:
            best_f1 = val_metrics["f1_ai"]
            torch.save(
                {
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "model_dir": args.model_dir,
                    "train_text_modes": train_modes,
                    "max_len": args.max_len,
                    "threshold": args.threshold,
                    "extra_feature_dim": len(train_examples[0].features),
                    "mlp_hidden_dim": args.mlp_hidden_dim,
                    "dropout": args.dropout,
                    "fusion_alpha_init": args.fusion_alpha_init,
                    "scaler": scaler.state_dict(),
                    "feature_names": TextFeatureEngineer.feature_names(),
                    "use_perplexity": args.use_perplexity and bool(args.ppl_model_dir),
                    "ppl_model_dir": args.ppl_model_dir,
                    "ppl_max_len": args.ppl_max_len,
                    "lora": lora_meta,
                },
                best_ckpt,
            )

    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best val f1_ai: {best_f1:.4f}")


def load_hybrid_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
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


def predict(args) -> None:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = pick_device(gpu_ids)
    model, tokenizer, scaler, ckpt = load_hybrid_checkpoint(args.checkpoint, device)
    feature_engineer = TextFeatureEngineer(
        use_perplexity=bool(ckpt.get("use_perplexity", False)),
        ppl_model_dir=ckpt.get("ppl_model_dir", ""),
        ppl_max_len=int(ckpt.get("ppl_max_len", 512)),
        device=device,
    )

    rows = load_csv_rows(args.csv_path)
    pred_modes = normalize_text_modes([args.text_mode] if args.text_mode else ckpt.get("train_text_modes", ["qa_concat"]))
    mode = pred_modes[0]
    feature_cache = FeatureDiskCache(args.feature_cache_db) if args.use_feature_cache else None
    examples = build_examples(
        rows,
        [mode],
        feature_engineer,
        require_label=False,
        feature_cache=feature_cache,
        feature_version=args.feature_cache_version,
    )
    if feature_cache is not None:
        feature_cache.close()
    for e in examples:
        e.features = scaler.transform(e.features)

    dataloader = DataLoader(
        TextDataset(examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, args.max_len or int(ckpt.get("max_len", 768))),
    )

    threshold = args.threshold if args.threshold is not None else float(ckpt.get("threshold", 0.5))
    preds: List[Tuple[float, int]] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)["logits"]
            probs = torch.sigmoid(logits).view(-1).cpu().tolist()
            preds.extend((float(p), int(p >= threshold)) for p in probs)

    for i, (prob, label) in enumerate(preds[: max(args.preview, 0)]):
        print(f"[{i}] prob_ai={prob:.4f}, pred={label}")

    if args.output_csv:
        fieldnames = list(rows[0].keys()) + ["prob_ai", "pred_label"]
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, (prob, label) in zip(rows, preds):
                out = dict(row)
                out["prob_ai"] = f"{prob:.6f}"
                out["pred_label"] = str(label)
                writer.writerow(out)
        print(f"Saved predictions to: {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Fine-tune Desklib detector with fused handcrafted features")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--model_dir", type=str, default="/mnt/jfzn/zjh/playground/ai-text-detector-v1.01")
    p_train.add_argument("--csv_path", type=str, default="")
    p_train.add_argument("--csv_paths", nargs="+", default=None)
    p_train.add_argument("--train_csv", type=str, default="")
    p_train.add_argument("--train_csvs", nargs="+", default=None)
    p_train.add_argument("--val_csv", type=str, default="")
    p_train.add_argument("--val_csvs", nargs="+", default=None)
    p_train.add_argument("--val_ratio", type=float, default=0.1)
    p_train.add_argument("--output_dir", type=str, required=True)
    p_train.add_argument("--train_text_modes", nargs="+", choices=["qa_concat", "a_only"], default=["qa_concat"])
    p_train.add_argument("--val_text_modes", nargs="+", choices=["qa_concat", "a_only"], default=None)
    p_train.add_argument("--max_len", type=int, default=768)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=0.01)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--fusion_alpha_init", type=float, default=-1.5)
    p_train.add_argument("--mlp_hidden_dim", type=int, default=256)
    p_train.add_argument("--threshold", type=float, default=0.5)
    p_train.add_argument("--max_grad_norm", type=float, default=1.0)
    p_train.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--use_perplexity", action=argparse.BooleanOptionalAction, default=False)
    p_train.add_argument("--ppl_model_dir", type=str, default="")
    p_train.add_argument("--ppl_max_len", type=int, default=512)
    p_train.add_argument("--use_feature_cache", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--feature_cache_db", type=str, default="./modeling/feature_cache.sqlite")
    p_train.add_argument("--feature_cache_version", type=str, default="v1")
    p_train.add_argument("--smoke_rows_per_csv", type=int, default=0)
    p_train.add_argument("--gpu_ids", type=str, default="")
    p_train.add_argument("--data_parallel", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--seed", type=int, default=42)
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

    p_predict = subparsers.add_parser("predict")
    p_predict.add_argument("--checkpoint", type=str, required=True)
    p_predict.add_argument("--csv_path", type=str, required=True)
    p_predict.add_argument("--output_csv", type=str, default="")
    p_predict.add_argument("--text_mode", type=str, choices=["qa_concat", "a_only"], default="")
    p_predict.add_argument("--max_len", type=int, default=0)
    p_predict.add_argument("--batch_size", type=int, default=32)
    p_predict.add_argument("--threshold", type=float, default=None)
    p_predict.add_argument("--preview", type=int, default=5)
    p_predict.add_argument("--gpu_ids", type=str, default="")
    p_predict.add_argument("--use_feature_cache", action=argparse.BooleanOptionalAction, default=True)
    p_predict.add_argument("--feature_cache_db", type=str, default="./modeling/feature_cache.sqlite")
    p_predict.add_argument("--feature_cache_version", type=str, default="v1")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()