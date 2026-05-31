import os
# 1. 清空环境变量
for var in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
            "LOCAL_RANK", "NODE_RANK", "SLURM_PROCID", "SLURM_NTASKS"]:
    os.environ.pop(var, None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.device_count = lambda: 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, GPU count: {torch.cuda.device_count()}")

# ================= 路径配置 =================
MODEL_PATH = "/mnt/data/tangleyi/AI_Text_Detection/Deep_Sleep_Homework/models/Qwen3.5-0.8B"
TRAIN_CSV  = "/mnt/data/tangleyi/AI_Text_Detection/Deep_Sleep_Homework/datasets/train/merged_dataset.csv"
VAL_CSV    = "/mnt/data/tangleyi/AI_Text_Detection/Deep_Sleep_Homework/datasets/val/UCAS_AISAD_TEXT-val.csv"
OUTPUT_DIR = "/mnt/data/tangleyi/AI_Text_Detection/Deep_Sleep_Homework/outputs_full_ft"   # 全量微调专用输出目录
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# ================= 超参数 =================
BATCH_SIZE = 2            # 全量微调显存占用大，从2开始安全
LR         = 5e-6         # 全量微调学习率要低，5e-6 ~ 1e-5 较合适
EPOCHS     = 3
MAX_LEN    = 512
GRAD_ACCUM = 8            # 有效 batch size = 2*8 = 16，可根据显存调整
SAVE_STEPS = 1000         # 每1000步保存一次checkpoint

# ================= 数据加载 =================
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["input_text"] = df["prompt"].astype(str) + "\n" + df["text"].astype(str)
    return df

train_df = load_csv(TRAIN_CSV)
val_df   = load_csv(VAL_CSV)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
pad_token_id = tokenizer.pad_token_id
print(f"Pad token id: {pad_token_id}")

class TextDataset(TorchDataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["input_text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = TextDataset(train_df, tokenizer, MAX_LEN)
val_dataset   = TextDataset(val_df, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================= 模型加载（无 LoRA） =================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=2, trust_remote_code=True
)

# 暴力设置 pad_token_id（避免 Qwen3.5 内部配置丢失）
def force_set_pad_token_id(module, pad_id):
    if hasattr(module, 'config') and hasattr(module.config, 'pad_token_id'):
        try:
            if module.config.pad_token_id is None:
                module.config.pad_token_id = pad_id
                print(f"   Set pad_token_id for {type(module).__name__}")
        except: pass
    for child in module.children():
        force_set_pad_token_id(child, pad_id)

force_set_pad_token_id(model, pad_token_id)
model.config.pad_token_id = pad_token_id
model.to(device)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ================= 优化器 & 学习率调度器 =================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# 断点续训
start_epoch = 0
best_metric = 0.0
checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_metric = checkpoint.get('best_metric', 0.0)
    print(f"Resumed training from epoch {start_epoch}, best_metric={best_metric:.4f}")
else:
    print("Starting training from scratch")

total_steps = (len(train_loader) // GRAD_ACCUM) * (EPOCHS - start_epoch)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1
)
if os.path.exists(checkpoint_path) and 'scheduler_state_dict' in checkpoint:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# ================= 评估函数 =================
def evaluate(model, val_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()          # 转 float32 防 bfloat16 报错
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, f1, auc

# ================= 日志与目录准备 =================
log_file = os.path.join(OUTPUT_DIR, "training_log.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
if start_epoch == 0:
    with open(log_file, "w") as f:
        f.write("epoch,step,loss_smoothed,val_acc,val_f1,val_auc,weighted\n")

# ================= 训练循环 =================
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    for step, batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        actual_loss = loss.item() * GRAD_ACCUM
        total_loss += actual_loss
        running_loss = 0.9 * running_loss + 0.1 * actual_loss
        pbar.set_postfix({"loss": f"{running_loss:.4f}"})

        if step % 10 == 0:
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{step},{running_loss:.6f},,,,\n")

        # 定期保存 checkpoint
        if (step + 1) % SAVE_STEPS == 0:
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric
            }, checkpoint_path)

    avg_loss = total_loss / len(train_loader) * GRAD_ACCUM
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # 评估
    acc, f1, auc = evaluate(model, val_loader)
    weighted = 0.6 * auc + 0.3 * acc + 0.1 * f1
    print(f"Val => Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Weighted: {weighted:.4f}")

    with open(log_file, "a") as f:
        f.write(f"{epoch+1},eval,,{acc:.4f},{f1:.4f},{auc:.4f},{weighted:.4f}\n")

    # 保存 epoch 结束 checkpoint
    torch.save({
        'epoch': epoch,
        'step': len(train_loader) - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1} end")

    # 保存最佳模型
    if weighted > best_metric:
        best_metric = weighted
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✓ Saved best model to {OUTPUT_DIR} (weighted={weighted:.4f})")

print("Training done.")