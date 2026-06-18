# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

微信聊天记录 → LoRA 微调数据管道。354 张超长竖屏聊天截图已提取为结构化 JSON，产出 ShareGPT 格式训练数据，用于微调对话模型学习"对方"的聊天风格。

**角色映射**：`me`（喵喵）= human 输入，`them`（阿狸/河狸/小河狸/凡凡）= gpt 学习目标。

**项目根目录**：`E:\狸喵galgame\`
**虚拟环境**：`E:\狸喵galgame\galgame\venv\`（Python 3.11）

## 目录结构

```
E:\狸喵galgame\
  */                            ← ~56 个分类文件夹，各含长截图 JPG（原始数据）
  sessions/                     ← 354 个会话 JSON（已产出，中间产物）
  results/                      ← API 分段提取中间结果（断点续传用）
  splitted/                     ← 统一分段的 JPG 缓存
  lora_data/                    ← 最终输出：ShareGPT 训练数据
    train.json / valid.json / dataset_info.json / stats.json
  galgame/                      ← 开发脚本和 venv
    venv/                       ← Python 3.11 虚拟环境
    split_v2.py                 ← 长截图切片工具
  extract_all.py                ← 批量提取主脚本（已跑完，无需重跑）
  build_lora_data.py            ← LoRA 数据管道（v1，当前使用）
  build_lora_data_v2.py         ← 人格化 system prompt 实验版（废弃，参考用）
  personas.json                 ← 角色人格定义（v2 实验产物，v1 不使用）
  colab_lora_train.ipynb        ← Colab 训练 Notebook
  .moonshot_key / .deepseek_key ← API Key 文件（用完需删除）
```

## 常用命令

```bash
# 构建训练数据（纯本地，秒级）
cd E:\狸喵galgame
./galgame/venv/Scripts/python build_lora_data.py

# 切分新图片
cd E:\狸喵galgame\galgame
python split_v2.py
```

Bash 下 `python` 需用 venv 完整路径：`./galgame/venv/Scripts/python`

## 数据格式

### Session JSON (sessions/)

```json
{"category":"0107道士下山", "source_image":"20220107_122955.jpg",
 "messages":[
   {"sender":"me"|"them","text":"...","time":"12:29",
    "type":"text|emoji|image|sticker|voice",
    "meta":{"emotion":"sarcastic|teasing|affectionate|neutral"}}
]}
```

- 18/354 个 sessions 有 TXT 黄金 emotion 标注，其余已剥离 emotion 字段
- `[sticker:xxx]` 是图片描述占位符，无法还原为原始图片

### ShareGPT (lora_data/)

```json
[{"source":"0107道士下山/20220107_122955.jpg", "conversations":[
  {"from":"system","value":"你正在微信上与朋友聊天。"},
  {"from":"human","value":"me的消息（上下文）"},
  {"from":"gpt","value":"them的回复（学习目标）"}
]}]
```

- system 是 conversations 内部的角色，不是顶层字段
- 每条样本以 human 开头（已过滤 gpt 开头的无效样本）

## 核心脚本

| 脚本 | 用途 | 状态 |
|------|------|:---:|
| `build_lora_data.py` | sessions → turns → 话题分割 → ShareGPT | ✅ 当前使用 |
| `build_lora_data_v2.py` | 同上 + 人格化 system prompt | ❌ 废弃（导致训练退化） |
| `extract_all.py` | 扫描目录→切片→Kimi Vision 提取→sessions | ✅ 已跑完 |
| `split_v2.py` | 单目录长截图切片 | ✅ 备用 |
| `colab_lora_train.ipynb` | Colab T4 LoRA 训练 | ✅ |

## 数据管道参数

- `TIME_GAP_MINUTES = 30`：话题分割时间阈值
- `CONTEXT_TURNS = 10`：gpt 回复的上下文窗口
- `TRAIN_RATIO = 0.8`：训练/验证集按日期切分

## 训练配置

### Colab（已验证）

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-3B-Instruct |
| 微调 | LoRA (rank=8, target=all linear) |
| 精度 | FP16（T4 不支持 BF16） |
| Batch | 2 × 8 梯度累积 = 16 |
| 学习率 | 5e-5 (cosine decay, warmup_ratio=0.1) |
| Epochs | 2 |
| 框架 | LLaMA-Factory (`llamafactory-cli train --template qwen`) |
| 耗时 | ~1h (6.4s/step, 574 steps) |
| 可训参数 | 14.97M / 3100M (0.48%) |

### LLaMA-Factory 关键点

- 入口是 `llamafactory-cli train`，**不是** `python src/train_bash.py`
- `dataset_info.json` 中 system 用 `tags.system_tag: "system"`，**不能**用 `columns.system`
- Qwen 家族必须加 `--template qwen`
- 3B FP16 ≈ 10GB，T4 16GB 够用，**不需要** `--quantization_bit 4`

## 训练历史

| 版本 | system prompt | 模型 | 结果 |
|------|:---|:---|------|
| v1 | `你正在微信上与朋友聊天。`（12 字） | 3B Colab | ✅ loss 5.57→3.21，风格可辨 |
| v2 | `你是阿狸...`（120 字完整人格） | 3B Colab | ❌ 退化：输出 sticker、角色混乱 |
| v3 | 同 v1 | 7B AutoDL | ✅ loss 3.39，37min，效果≈v1 |

**v2 失败原因**：
- 120 字 system prompt 重复 4592 条 → 稀释对话信号
- `cutoff_len=1024` 下 prompt 吃掉 60 tokens，上下文不足
- "少用emoji"指令与训练数据中 `[sticker:xxx]` 矛盾

**教训**：
- 训练时 system prompt 应极简（≤20 字），风格从数据中自然学习
- 推理时使用带身份的人格化 prompt（`你是阿狸...`）
- **7B vs 3B 无显著差异** — 风格迁移瓶颈在数据，不在模型大小

## 推理策略

**训练用泛用 prompt，推理用时人格化 prompt**（已验证最优）：

```python
# 训练时（数据里）
system = "你正在微信上与朋友聊天。"  # 12 字

# 推理时（代码里）
system = "你是阿狸，一个爱撒娇、喜欢黏着喵喵的女生。你正在和男朋友喵喵微信聊天。回复短小精炼，爱撒娇。"
```

测试脚本：`test_lora_batch.py`（AutoDL 批量测试）、`colab_lora_train.ipynb` Step 6（Colab 交互测试）。

## API 使用

- **Kimi (Moonshot)**：视觉提取，`moonshot-v1-8k-vision-preview`，200 RPM，¥19/354 图
- **DeepSeek**：纯文本分类，`deepseek-chat`
- 两者兼容 OpenAI SDK，改 `base_url` 即可
- 训练零 API 调用，Key 文件不上传云服务器

## Known Issues

1. `split_v2.py`：`bottom >= h` 时需立即 `break`，否则死循环
2. Kimi 文本模型高峰期 429 → 用 DeepSeek 替代或指数退避
3. Emotion 标注准确率 ~57%（DeepSeek 纯文本），优先用 TXT 黄金标注
4. Cloud GPU 无 VPN 拉模型：
   - `export HF_ENDPOINT=https://hf-mirror.com`（HuggingFace 镜像）
   - 或用 ModelScope 国内源下载：`pip install modelscope && python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models')"`
5. AutoDL 按开机计费，训完立即关机
6. AutoDL 无外网：`pip install` 失败 → 7B 不加 `--quantization_bit 4` 直接跑
7. LLaMA-Factory v0.9+ 要求 conversations 中 system 后第一条必须是 human → `build_lora_data.py` 已过滤
8. `dataset_info.json` 中 system 用 `tags.system_tag`，不能用 `columns.system`
9. AutoDL 终端输入中文后删除字符 → 异常空格导致 tokenizer 崩溃 → 用批量脚本来测试
10. 下载 LoRA 只需 `adapter_model.safetensors` + `adapter_config.json`（~40MB），不需要整个 output（700MB）
