# GalgameLoRA

微信聊天风格 LoRA 微调 — 数据、脚本、权重一站式仓库。

## 快速开始

### 训练

```bash
# Colab（3B 免费 T4）
# 上传 data/ 到 Google Drive，运行 training/colab_lora_train.ipynb

# AutoDL（7B 4090D）
cd /root/LLaMA-Factory
LLAMAFACTORY_LOGGING_LEVEL=warning
llamafactory-cli train \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --dataset chat_style_train \
  --output_dir /root/autodl-tmp/output/chat_style_lora_7b \
  --stage sft --do_train --finetuning_type lora --template qwen \
  --lora_rank 8 --lora_target all \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 --num_train_epochs 2 \
  --lr_scheduler_type cosine --warmup_ratio 0.1 --cutoff_len 1024 \
  --bf16 --overwrite_output_dir
```

### 推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", ...)
model = PeftModel.from_pretrained(model, "weights/3b")

SYSTEM_PROMPT = "你是阿狸，一个爱撒娇、喜欢黏着喵喵的女生。你正在和男朋友喵喵微信聊天。回复短小精炼，爱撒娇。"
```

### 构建数据

```bash
python scripts/build_sharegpt_v2.py
```

## 目录

| 目录 | 内容 |
|------|------|
| `data/` | ShareGPT 训练数据（9,366 train + 2,342 valid） |
| `scripts/` | 数据构建脚本 |
| `training/` | Colab Notebook + AutoDL 推理脚本 |
| `weights/` | 3B / 7B LoRA 权重（adapter 文件） |
| `docs/` | 训练指南、踩坑记录、复现文档 |

## 训练结果

| 版本 | 模型 | Loss | 说明 |
|------|------|------|------|
| v1 | Qwen2.5-3B-Instruct | 5.57→3.21 | Colab T4, 1h |
| v3 | Qwen2.5-7B-Instruct | 3.39 | AutoDL 4090D, 37min |

## 关键策略

- **训练用泛用 prompt**：`你正在微信上与朋友聊天。`（12 字）
- **推理用人格 prompt**：注入角色身份，风格从数据自然学习
- 3B 足够，7B 无显著提升 — 瓶颈在数据质量，不在模型大小
