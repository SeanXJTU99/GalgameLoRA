# GGUF 端侧部署指南

> 将 v4 LoRA 权重 merge → GGUF → Q4_K_M 静态量化，分发到手机（Android ChatterUI / PocketPal / iOS LLMFarm）。

## 背景

| 方案 | bitsandbytes 动态 4bit（已有） | GGUF Q4_K_M（本文） |
|------|:---:|:---:|
| 量化时机 | 运行时 | 离线静态 |
| 产物 | 无独立文件 | ~4.7GB .gguf 文件 |
| 推理框架 | LLaMA-Factory | llama.cpp / Ollama / 手机 app |
| 校准数据 | 不需要 | 不需要 |
| 平台 | PC（1080Ti 等） | PC + 手机 + WebGPU |

`低显存无损推理.md` 解决了 PC 端显存瓶颈，本文解决跨平台分发。

## 流程

全部在 AutoDL 4090D 上完成。

### 1. Merge LoRA → fp16 完整模型

```bash
llamafactory-cli export \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --adapter_name_or_path /root/autodl-tmp/output/chat_style_lora_7b_pure \
  --template qwen \
  --export_dir /root/autodl-tmp/output/qwen2.5_7b_merged \
  --export_legacy_format False
```

### 2. 安装 llama.cpp（Python 绑定 + 工具链）

```bash
pip install llama-cpp-python

# 编译量化工具
cd /root/autodl-tmp
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build llama.cpp
cmake --build llama.cpp/build -t llama-quantize -j
```

### 3. 转 f16 GGUF

```bash
# 方式 A：llama-cpp-python（推荐，纯 Python）
python -m llama_cpp.convert \
  /root/autodl-tmp/output/qwen2.5_7b_merged \
  --outtype f16 \
  --outfile /root/autodl-tmp/output/qwen7b_chatstyle_f16.gguf

# 方式 B：llama.cpp 原始脚本
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python llama.cpp/convert_hf_to_gguf.py \
  /root/autodl-tmp/output/qwen2.5_7b_merged \
  --outfile /root/autodl-tmp/output/qwen7b_chatstyle_f16.gguf \
  --outtype f16
```

### 4. Q4_K_M 量化

```bash
./llama.cpp/build/bin/llama-quantize \
  /root/autodl-tmp/output/qwen7b_chatstyle_f16.gguf \
  /root/autodl-tmp/output/qwen7b_chatstyle_Q4_K_M.gguf \
  Q4_K_M
```

耗时约 5-10 分钟。产物大小：

| 格式 | 大小 | 说明 |
|------|------|------|
| merged fp16 | ~15 GB | merge 后原始文件夹 |
| f16 GGUF | ~15 GB | 中间产物，量化后可删 |
| **Q4_K_M** | **~4.7 GB** | 最终分发文件 |

### 5. 验证推理

```bash
# 非交互
llama-cli -m /root/autodl-tmp/output/qwen7b_chatstyle_Q4_K_M.gguf \
  -p "你是个爱撒娇的女孩，正在和男朋友聊天\n\nUser: 想你了\nAssistant:" \
  -n 128 --temp 0.7

# 交互模式
llama-cli -m /root/autodl-tmp/output/qwen7b_chatstyle_Q4_K_M.gguf \
  --chat-template chatml \
  -n 256 --temp 0.7
```

## 手机端加载

### Android

- [ChatterUI](https://github.com/Vali-98/ChatterUI) — 内置 llama.cpp，直接加载 GGUF
- [PocketPal AI](https://github.com/a-ghorbani/PocketPal) — 最简上手，支持多模型
- [Layla](https://github.com/Layla-Network/Layla) — 本地 AI 伴侣

### iOS

- [LLMFarm](https://github.com/guinmoon/LLMFarm) — llama.cpp iOS 封装
- PocketPal AI 也有 iOS 版

使用方式：将 `.gguf` 文件传到手机，app 内指定路径加载即可。

### System Prompt 设置

推理时使用长版人格化 prompt（与训练短版区分，见 README 关键策略）：

```
你是阿狸，一个爱撒娇、喜欢黏着喵喵的女生。说话带叠词、语气词、颜文字和 sticker 表情包。
```

## 量化精度对比

| 量化级别 | 大小 | 质量 | 手机推荐 |
|----------|------|------|:---:|
| Q4_0 | ~4.3 GB | 可接受 | ✓ ARM 优化好 |
| **Q4_K_M** | **~4.7 GB** | **好** | ✓ 推荐首选 |
| Q5_K_M | ~5.7 GB | 更好 | △ 6GB 内存可冲 |
| Q8_0 | ~8.3 GB | 近无损 | ✗ 手机吃力 |

## 备选：3B 模型

如果手机跑 7B 太慢/内存不够，可取 `weights/3b/`（v1 LoRA），同流程走一遍：

- Qwen2.5-3B-Instruct + v1 LoRA → merge → Q4_K_M ≈ **2.0 GB**
- 更省资源，但效果不及 v4（Loss 3.21 vs 1.06）

或者用 v4 配方在 Qwen2.5-3B-Instruct 上重训 → merge → GGUF（详见 `TRAINING_LOG_v4.md` 超参）。

## 常见问题

- **Q4_K_M 和 bitsandbytes 4bit 哪个好？** Q4_K_M 略优（分组量化），且是静态文件不依赖 bitsandbytes。
- **需要校准数据吗？** Q4_K_M 不需要。GPTQ/AWQ 需要校准数据集（这也是之前 LLaMA-Factory export 报错的原因）。
- **Ollama 怎么用？** 写 Modelfile：`FROM qwen7b_chatstyle_Q4_K_M.gguf` + `ollama create chatstyle --modelfile Modelfile`。
