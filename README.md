# GalgameLoRA

微信聊天风格 LoRA 微调 → 陪伴 Agent。数据、脚本、权重、方案一站式仓库。

**当前阶段**：v4 风格引擎已完成（loss=1.06），向陪伴 Agent 演进中。

## 快速开始

### 训练

```bash
# AutoDL（7B 4090D）
cd /root/LLaMA-Factory
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
# 方式 1：LoRA + bitsandbytes 4-bit（1080Ti，已验证）
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path weights/7b_v4 \
  --template qwen --export_dir ./merged
llamafactory-cli chat --model_name_or_path ./merged --template qwen --quantization_bit 4

# 方式 2：GGUF Q4_K_M + llama-server（推荐，生产用，1080Ti）
# 详见 docs/GGUF端侧部署指南.md（合并+量化步骤）
llama-server -m qwen7b_chatstyle_Q4_K_M.gguf --n-gpu-layers 99
```

详见 `docs/低显存无损推理.md`。

### 构建数据

```bash
python scripts/build_sharegpt_v2.py
```

## 目录

| 目录 | 内容 |
|------|------|
| `data/` | ShareGPT 训练数据（隐私，需本地构建） |
| `scripts/` | 数据管道（build_sharegpt_v2.py 当前使用） |
| `training/` | Colab Notebook + 推理测试脚本 |
| `weights/3b/` | v1 3B LoRA |
| `weights/7b/` | v3 7B LoRA（rank=8） |
| `weights/7b_v4/` | **v4 7B LoRA（rank=32，最优）** |
| `docs/` | 训练指南、踩坑记录、技术调研、后续方案 |

## 训练结果

| 版本 | 模型 | rank | lr | epochs | Loss | 说明 |
|------|------|------|-----|--------|------|------|
| v1 | 3B | 8 | 5e-5 | 2 | 3.21 | Colab T4, 1h |
| v3 | 7B | 8 | 5e-5 | 2 | 3.39 | AutoDL 4090D, 37min |
| **v4** | **7B** | **32** | **2e-4** | **4** | **1.06** | **AutoDL 4090D, 1h17min** |

## 关键策略

- **训练用短 prompt，推理用长 prompt**：训练时 `你是个爱撒娇的女孩，正在和男朋友聊天`（18 字），推理时用人格化 prompt
- **合并同 sender + 滑动窗口**：保证 human/gpt 交替，MAX_CTX=8
- **异构分发**：认知用 API（GPT-4o-mini），风格用 LoRA 7B（1080Ti GGUF），嵌入用本地小模型（BGE+RoBERTa）

## 后续路线

```
风格引擎 ✓(v4) → 推理服务化 → 记忆增强 → 情绪关系 → 偏好优化
```

详见 `docs/后续方案.md`、`docs/技术选型审查.md`、`docs/情感陪伴AI技术路线调研_2025-2026.md`。
