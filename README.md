# AI文本检测模型 - 课程设计

这是一个用于检测AI生成文本的分类模型，基于Qwen3-Embedding-0.6B和LoRA微调。

## 环境准备

首先，请确保您的系统已安装Python 3.8+。然后安装所需依赖：

```bash
pip install -r requirements.txt
```

## 项目结构

```
.
├── README.md               # 项目说明文档
├── requirements.txt        # 项目依赖包列表

├── modeling/               # 核心代码目录
│   ├── train_qwen_ai_detector.py  # 训练和推理的主脚本
│   ├── checkpoints/        # 模型权重保存路径（文件未上传）
│   └── README.md           # 脚本使用说明

├── datasets/               # 数据集存放目录
    ├── train/merged_dataset.csv  # 训练数据集
    └── ...                 # 其他数据集

└── comparing solutions     # 其他方案

```

## 快速开始

### 1. 下载模型权重

由于模型文件较大，未上传至GitHub。请从以下链接下载预训练模型权重：

[上传至Modelscope平台的预训练模型](https://modelscope.cn/models/FAndromedA/qwen_0.6B_ai_detector/summary)

下载后，将模型文件（例如 `best_qwen_ai_detector.pt`）放入 `./modeling/checkpoints/` 目录。

### 2. 运行推理

使用 `modeling/train_qwen_ai_detector.py` 脚本进行推理，生成符合官方要求的Excel提交文件。

```bash
python modeling/train_qwen_ai_detector.py predict_csv \
    --checkpoint ./modeling/checkpoints/[你的模型文件名].pt \
    --input_csv ./datasets/test1/UCAS_AISAD_TEXT-test1.csv \
    --output_csv ./outputs/[你的队伍名].xlsx
```

### 3. 提交结果

将生成的 `[你的队伍名].xlsx` 文件提交给老师。
