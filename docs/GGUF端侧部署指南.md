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

### 0. 环境准备（新实例执行一次）

```bash
# 0.1 venv 建在数据盘（系统盘仅 30G，pip 全装必炸）
python -m venv /root/autodl-tmp/venv --system-site-packages
source /root/autodl-tmp/venv/bin/activate

# 0.2 安装 LLaMA-Factory
pip install llamafactory -i https://pypi.tuna.tsinghua.edu.cn/simple

# 0.3 底座模型路径（AutoDL 预装选 Qwen2.5-7B-Instruct 即可跳过 modelscope）
# ls Qwen2.5/text-generation-webui/models/Qwen2.5-7B-Instruct/

# 0.4 上传 LoRA 权重（本地上传或用 scp）
# 将 weights/7b_v4/adapter_config.json + adapter_model.safetensors 传到
# /root/autodl-tmp/output/chat_style_lora_7b_pure/
```

### 1. Merge LoRA → fp16 完整模型

```bash
# 底座路径以 AutoDL 预装为准，常见位置：
# - 预装选 Qwen2.5-7B-Instruct → Qwen2.5/text-generation-webui/models/Qwen2.5-7B-Instruct
# - 手动 modelscope 下载 → /root/autodl-tmp/models/Qwen2.5-7B-Instruct

llamafactory-cli export \
  --model_name_or_path Qwen2.5/text-generation-webui/models/Qwen2.5-7B-Instruct \
  --adapter_name_or_path /root/autodl-tmp/output/chat_style_lora_7b_pure \
  --template qwen \
  --export_dir /root/autodl-tmp/output/qwen2.5_7b_merged \
  --export_legacy_format False
```

### 2. 安装 llama.cpp（Python 绑定 + 工具链）

```bash
cd /root/autodl-tmp
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build llama.cpp
cmake --build llama.cpp/build -j  # 全编（含 llama-cli + llama-quantize）
```

### 3. 转 f16 GGUF

```bash
# 安装转换依赖（会降级 tokenizers/transformers/torch，与 llamafactory 冲突但无影响）
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 转换（约 10-15 分钟）
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
# llama-cli 编译后不在系统 PATH，需用完整路径
LLAMA_CLI=/root/autodl-tmp/llama.cpp/build/bin/llama-cli

# 交互模式
$LLAMA_CLI -m /root/autodl-tmp/output/qwen7b_chatstyle_Q4_K_M.gguf --chat-template chatml -n 256 --temp 0.7
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

## 踩坑记录

### 坑 1：系统盘满 (Errno 28 No space left on device)

**现象**：`pip install llamafactory` 报 `OSError: [Errno 28] No space left on device`

**原因**：AutoDL 系统盘 30G，pip cache + 依赖装系统盘直接满。

**解决**：
```bash
pip cache purge
python -m venv /root/autodl-tmp/venv --system-site-packages
source /root/autodl-tmp/venv/bin/activate
pip install llamafactory -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 坑 2：llama-cli: command not found

**现象**：编译完 `cmake --build llama.cpp/build -t llama-quantize` 后，运行 `llama-cli` 报 command not found。

**原因**：①只编译了 `llama-quantize` 一个 target，没编 `llama-cli`。②即便编了，`llama.cpp/build/bin/` 也不在 `$PATH` 中。

**解决**：
```bash
# 全编所有工具
cmake --build llama.cpp/build -j

# 用完整路径
/root/autodl-tmp/llama.cpp/build/bin/llama-cli -m ...
```

### 坑 3：pip 依赖冲突警告（不影响结果）

**现象**：安装 `requirements-convert_hf_to_gguf.txt` 后，`pip check` 报 `llamafactory` 与 `tokenizers`/`transformers` 版本冲突。

**原因**：GGUF 转换脚本需求新版 transformers，会被降级安装。

**结论**：忽略。LLaMA-Factory 已完成 merge，后续不再需要它。GGUF 转换和量化正常执行。

### 坑 4：LoRA 必须两个文件

**现象**：只上传 `adapter_model.safetensors`，merge 报错找不到 adapter 配置。

**解决**：`adapter_config.json` + `adapter_model.safetensors` 两个文件都要传，放在同一目录。

### 坑 5：底座路径不是绝对路径自动查找

**现象**：`--model_name_or_path Qwen2.5-7B-Instruct` 报 model not found。

**原因**：AutoDL 预装的底座不在 huggingface cache，需要指定实例上的实际路径。

**解决**：`--model_name_or_path Qwen2.5/text-generation-webui/models/Qwen2.5-7B-Instruct`（或 `ls` 确认实际位置）。

## 常见问题

- **Q4_K_M 和 bitsandbytes 4bit 哪个好？** Q4_K_M 略优（分组量化），且是静态文件不依赖 bitsandbytes。
- **需要校准数据吗？** Q4_K_M 不需要。GPTQ/AWQ 需要校准数据集（这也是之前 LLaMA-Factory export 报错的原因）。
- **Ollama 怎么用？** 写 Modelfile：`FROM qwen7b_chatstyle_Q4_K_M.gguf` + `ollama create chatstyle --modelfile Modelfile`。
