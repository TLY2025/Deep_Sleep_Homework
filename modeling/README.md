解冻最后两层，训练。

ISIBLE_DEVICES=4 nohup python modeling/train_qwen_ai_detector.py train \
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

lora 训练。

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

csv 推理

nohup python modeling/train_qwen_ai_detector.py predict_csv \
    --checkpoint /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/checkpoints/qwen_ai_detector_lora/epoch_1_qwen_ai_detector.pt \
    --input_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/datasets/test1/UCAS_AISAD_TEXT-test1.csv \
    --output_csv /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/outputs/qwen_ai_detector_lora_predict.csv \
    --max_len 2048 \
    --batch_size 248 \
    --human_threshold 0.5 \
    --dtype bfloat16 \
    --gpu 5  > /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/logs/qwen_ai_detector_lora_predict.log 2>&1 &

单条文本推理：

conda run -n zjh_mm python modeling/train_qwen_ai_detector.py predict_text \
  --checkpoint /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/modeling/checkpoints/qwen_detector_v1/best_qwen_ai_detector.pt \
  --text "这里是一段待检测文本"
QA 对推理：

conda run -n zjh_mm python modeling/train_qwen_ai_detector.py predict_text \
  --checkpoint /mnt/jfzn/zjh/playground/Deep_Sleep_Homework/modeling/checkpoints/qwen_detector_v1/best_qwen_ai_detector.pt \
  --question "问题" \
  --answer "回答"