# GalgameLoRA

微信聊天风格 LoRA 微调 — 数据、脚本、权重一站式仓库。

## 快速开始

### 训练

```bash
# Colab（3B 免费 T4）
# 上传 data/ 到 Google Drive，运行 training/colab_lora_train.ipynb

# AutoDL（7B 4090D，激进策略）
cd /root/LLaMA-Factory
LLAMAFACTORY_LOGGING_LEVEL=warning
llamafactory-cli train \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --dataset chat_style_train \
  --output_dir /root/autodl-tmp/output/chat_style_lora_7b \
  --stage sft --do_train --finetuning_type lora --template qwen \
  --lora_rank 32 --lora_alpha 64 --lora_target all \
  --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 --num_train_epochs 4 \
  --lr_scheduler_type cosine --warmup_ratio 0.1 --cutoff_len 2048 \
  --bf16 --overwrite_output_dir
```

### 推理

```bash
# 方式 1：LoRA 直接推理（需 GPU）
python training/test_lora_batch.py

# 方式 2：Merge + 4bit 量化推理（低显存，1080Ti 可用）
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path weights/7b_v4 \
  --template qwen \
  --export_dir ./merged

llamafactory-cli chat \
  --model_name_or_path ./merged \
  --template qwen \
  --quantization_bit 4
```

详见 `docs/低显存无损推理.md`。

#### 方式 3：GGUF 端侧部署（手机/跨平台）

```bash
# 1. Merge 同上
# 2. 转 f16 GGUF
python -m llama_cpp.convert ./merged --outtype f16 --outfile ./f16.gguf
# 3. Q4_K_M 量化
llama-quantize ./f16.gguf ./q4km.gguf Q4_K_M
# 4. 手机 app（ChatterUI/PocketPal/LLMFarm）加载 q4km.gguf
```

详见 `docs/GGUF端侧部署指南.md`。

### 构建数据

```bash
python scripts/build_sharegpt_v2.py
```

## 目录

| 目录 | 内容 |
|------|------|
| `data/` | 训练数据（4,796 train + 1,199 valid，隐私数据不公开，需本地构建） |
| `scripts/` | 数据构建脚本（build_sharegpt_v2.py 当前使用） |
| `training/` | Colab Notebook + 推理脚本 |
| `weights/3b/` | v1 3B LoRA 权重 |
| `weights/7b/` | v3 7B LoRA 权重（rank=8） |
| `weights/7b_v4/` | v4 7B LoRA 权重（rank=32，最优） |
| `docs/` | 训练指南、踩坑记录、复现文档 |

## 训练结果

| 版本 | 模型 | rank | lr | epochs | Loss | 数据 | 说明 |
|------|------|------|-----|--------|------|------|------|
| v1 | 3B | 8 | 5e-5 | 2 | 3.21 | v1 5.7K | Colab T4, 1h |
| v3 | 7B | 8 | 5e-5 | 2 | 3.39 | v1 5.7K | AutoDL 4090D, 37min |
| **v4** | **7B** | **32** | **2e-4** | **4** | **1.06** | **v2 6.0K** | **AutoDL 4090D, 1h17min** |

## 关键策略

- **训练用短身份 prompt**：`你是个爱撒娇的女孩，正在和男朋友聊天`
- **推理用人格化 prompt**：`你是阿狸，一个爱撒娇、喜欢黏着喵喵的女生...`
- **合并同 sender + 滑动窗口**：保证 human/gpt 交替，每条样本 ≤10 条消息
- **Merge → 4bit**：合并后量化推理，显存 22G→6G，1080Ti 可跑
