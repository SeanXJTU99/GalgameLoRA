# v4 训练日志（7B 激进策略 + 量化推理）

> 2026-06-19 | Qwen2.5-7B-Instruct | AutoDL 4090D

## 数据

| 指标 | 值 |
|------|-----|
| 训练样本 | 4,796 |
| 验证样本 | 1,199（未加载 eval） |
| system prompt | `你是个爱撒娇的女孩，正在和男朋友聊天`（18 字） |
| 数据版本 | v2 净化版（合并同 sender、按日期拼接、滑动窗口 MAX_CTX=8） |

## 超参

| 参数 | v1/v3 保守 | v4 激进 |
|------|:---:|:---:|
| lora_rank | 8 | **32** |
| lora_alpha | — | **64** |
| lora_target | all | all |
| 学习率 | 5e-5 | **2e-4** |
| Epochs | 2 | **4** |
| Batch (有效) | 8-16 | 16 |
| cutoff_len | 1024 | 2048 |

## 训练命令

```bash
llamafactory-cli train \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --dataset chat_style_train \
  --output_dir /root/autodl-tmp/output/chat_style_lora_7b_pure \
  --stage sft \
  --do_train \
  --finetuning_type lora \
  --template qwen \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_target all \
  --gradient_checkpointing True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --cutoff_len 2048 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 \
  --overwrite_output_dir
```

## 结果

| 指标 | 值 |
|------|-----|
| Train Loss | **1.06** |
| 耗时 | 1h 17min |
| Epochs | 4.0 |
| 样本/秒 | 4.13 |
| 步/秒 | 0.257 |

## 对比

| 版本 | 模型 | system | rank | lr | epochs | Loss |
|------|------|--------|------|-----|--------|------|
| v1 | 3B | 通用 12 字 | 8 | 5e-5 | 2 | 3.21 |
| v3 | 7B | 通用 12 字 | 8 | 5e-5 | 2 | 3.39 |
| **v4** | **7B** | **撒娇女孩 18 字** | **32** | **2e-4** | **4** | **1.06** |

## 推理优化

LoRA 合并 → 4bit 量化，显存 22G → <6G，支持 1080Ti。

```bash
# 1. merge
llamafactory-cli export \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --adapter_name_or_path /root/autodl-tmp/output/chat_style_lora_7b_pure \
  --template qwen \
  --export_dir /root/autodl-tmp/output/qwen2.5_7b_merged \
  --export_legacy_format False

# 2. 4bit 推理
llamafactory-cli chat \
  --model_name_or_path /root/autodl-tmp/output/qwen2.5_7b_merged \
  --template qwen \
  --quantization_bit 4
```

详见 `低显存无损推理.md`。
