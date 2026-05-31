import time
import torch
from transformers import pipeline

# 你的模型路径
MODEL_PATH = "./chatgpt-qa-detector-roberta"

# 加载模型
start_load = time.time()
detector = pipeline(
    "text-classification",
    model=MODEL_PATH,
    device=0,
    torch_dtype=torch.float16
)
load_time = time.time() - start_load

# 测试文本
prompt = "What is AI?"
answer = "AI stands for Artificial Intelligence."
text = f"{prompt}</s>{answer}"

# ===================== 第一次推理（热身，必慢）
print("=== 第一次推理（热身）===")
start1 = time.time()
result = detector(text)[0]
time1 = time.time() - start1

# ===================== 第二次推理（真实速度，超快）
print("=== 第二次推理（正式，4090全速）===")
start2 = time.time()
result = detector(text)[0]
time2 = time.time() - start2

# 结果解析
label_map = {"LABEL_0": "人类撰写", "LABEL_1": "AI生成"}
final_label = label_map[result["label"]]

# 输出
print("="*60)
print(f"结果：{final_label}")
print(f"模型加载：{load_time:.2f}s")
print(f"第一次推理（热身）：{time1:.4f}s")
print(f"第二次推理（真实速度）：{time2:.4f}s")
print("="*60)

# acc 1. 1-0.1398 
# acc 2. 1-0.0378
# acc 3. 1-0.16
# acc 4. 1-0.0259