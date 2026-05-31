import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

# 使用 local_dir 参数指定下载路径
model_dir = snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    local_dir="./AI_Text_Detection/Deep_Sleep_Homework/models/Qwen3.5-0.8B", # 模型将被下载到这个目录
)

print(f"Model downloaded to: {model_dir}")