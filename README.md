# GalgameLoRA — WeChat Chat-Style Companion Agent

LoRA fine-tuned Qwen2.5-7B that mimics a girlfriend's chat style (pet names, stickers, stickers), plus a memory-augmented FastAPI agent. Trained on 354 real WeChat sessions (~4,800 samples), deployed as GGUF Q4_K_M on AutoDL on-demand GPU.

## Features

- **LoRA Style Engine (loss 1.06)**: Rank-32 LoRA on Qwen2.5-7B-Instruct, training-only short system prompt (18 chars) to avoid signal dilution
- **RLAIF → DPO Pipeline**: AI-judge generated preference pairs for RL-free alignment improvement
- **Memory-Augmented Agent**: ChromaDB + bge-small-zh-v1.5 retrieval with lazy-decay importance scoring
- **Dual-Route Context Assembly**: Rich (LLM-composed ≤200 chars) vs Minimal (persona + ≤30 chars) — config switch, not branch
- **Three-Backend Emotion Detection**: Rule-based (default), RoBERTa (lazy load), or LLM (side-output from orchestrator)
- **GGUF Q4_K_M Deployment**: Static 4-bit quantization, llama-server HTTP API, fits 8GB GPU VRAM

## Architecture

```
┌──────────────────────────┐
│  FastAPI Agent (local)   │  ← ChromaDB + BGE (24MB free)
│  POST /chat              │
├──────────────────────────┤
│  Orchestrator (DeepSeek) │  ← Memory extraction + context assembly (~¥0.001/round)
├──────────────────────────┤
│  Style Engine (AutoDL)   │  ← llama-server + GGUF (on-demand, ¥1.68/h)
│  Qwen2.5-7B Q4_K_M      │
└──────────────────────────┘
```

## Hardware & Cost

| Resource | Spec | Cost |
|----------|------|:---:|
| Training GPU | AutoDL 4090D | ¥1.68/h, ~1.3h/train |
| Inference GPU | AutoDL 4090D on-demand | ~¥30-60/month |
| Orchestrator LLM | DeepSeek-chat | ~¥0.001/round |
| Local CPU/RAM | i7 / 8GB | Runs Agent + ChromaDB + BGE |
| GGUF model | Q4_K_M | 4.7GB (local download) |

## Quick Start

### Training

```bash
# AutoDL 4090D
cd /root/LLaMA-Factory
llamafactory-cli train \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --dataset chat_style_train --template qwen \
  --output_dir /root/autodl-tmp/output/chat_style_lora_7b \
  --stage sft --do_train --finetuning_type lora \
  --lora_rank 32 --lora_alpha 64 --lora_target all \
  --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 --num_train_epochs 4 \
  --lr_scheduler_type cosine --warmup_ratio 0.1 --cutoff_len 2048 \
  --bf16 --overwrite_output_dir
```

### Merge & Quantize

```bash
# 1. Merge LoRA → fp16
llamafactory-cli export --model_name_or_path <base-model> \
  --adapter_name_or_path weights/7b_v4 --template qwen --export_dir ./merged

# 2. Convert → GGUF
python llama.cpp/convert_hf_to_gguf.py ./merged --outtype f16 --outfile f16.gguf

# 3. Q4_K_M quantize
./llama-quantize f16.gguf qwen7b_chatstyle_Q4_K_M.gguf Q4_K_M

# 4. Serve (AutoDL)
llama-server -m qwen7b_chatstyle_Q4_K_M.gguf --host 0.0.0.0 --port 6006 --ctx-size 2048
```

Full guide: `docs/GGUF端侧部署指南.md`

### Agent

```bash
# Install
cd agent && pip install -r requirements.txt

# Configure .env
LLAMA_SERVER_URL=https://<autodl-proxy>:8443
LLAMA_API_KEY=<key>
LLM_API_KEY=<deepseek-key>

# Run
uvicorn server:app --port 8000
```

Open `http://localhost:8000` for the chat UI.

### Build Data

```bash
python scripts/build_sharegpt_v2.py   # Requires sessions/ from parent project
```

### Run Tests

```bash
cd agent && python -m pytest tests -q   # 28 passed, no model download
```

## Training Pipeline

```
WeChat sessions (354 JSON)
  → build_sharegpt_v2.py (merge same-sender, sliding window MAX_CTX=8)
  → ShareGPT train.json (4,796) + valid.json (1,199)
  → Qwen2.5-7B-Instruct + LoRA rank=32 SFT (~1.3h 4090D)
  → v4 adapter (loss 1.06)
  → RLAIF preference generation (temp 0.7 vs 1.0, ~3h)
  → DeepSeek judge (3-dim scoring) → DPO pairs
  → DPO fine-tuning (~2h 4090D)
  → Merge → GGUF Q4_K_M (4.7GB)
  → llama-server HTTP API
```

## lora训练结果

| 版本 | 模型 | rank | lr | epochs | Loss | 说明 |
|------|------|------|-----|--------|------|------|
| v1 | 3B | 8 | 5e-5 | 2 | 3.21 | Colab T4, 1h |
| v3 | 7B | 8 | 5e-5 | 2 | 3.39 | AutoDL 4090D, 37min |
| **v4** | **7B** | **32** | **2e-4** | **4** | **1.06** | **AutoDL 4090D, 1h17min** |

## Project Structure

```
scripts/          Data pipeline (build_sharegpt_v2.py)
training/         Colab notebook + inference test scripts
weights/
  3b/             v1 3B LoRA adapter
  7b/             v3 7B LoRA adapter (rank=8)
  7b_v4/          v4 7B LoRA adapter (rank=32, best)
agent/
  server.py       FastAPI orchestrator
  config.py       Env config (CONTEXT_MODE, EMOTION_BACKEND, LLM_BASE_URL)
  memory/         ChromaDB store + LLM extraction
  emotion/        Rule / RoBERTa / LLM detector
  context/        Rich vs minimal assembler
  generation/     llama-server HTTP client
  frontend/       Single-file WeChat-style chat UI
  eval/           Style baseline + RLAIF generate/judge
  tests/          28 tests (offline, no model download)
docs/             GGUF guide, low-VRAM inference, training logs
data/             Training data (private, in .gitignore)
```

## 关键策略

- **训练用短 prompt，推理用长 prompt**：训练 `你是个爱撒娇的女孩，正在和男朋友聊天`（18 字），推理用人格化 120 字 prompt。v2 验证过长 prompt 导致退化
- **合并同 sender + 滑动窗口**：保证 human/gpt 严格交替，MAX_CTX=8
- **LoRA 不跨底座**：7B adapter 只能 merge 7B，shape 不匹配
- **GGUF 不需要校准数据**：绕过 LLaMA-Factory 导出 bug，用 llama.cpp 直接转
- **数据隐私**：`data/` 和 `.env` 均 `.gitignore`，不上传 GitHub
