# AutoDL 云端部署实操笔记

> 汇总于 2026-06-17 | 基于实际操作中遇到的问题

---

## 一、AutoDL 基础

### 磁盘

| 路径 | 类型 | 典型容量 | 关机保留 |
|------|------|---------|:---:|
| `/root/` | 系统盘 | ~50GB | ❌ |
| `/root/autodl-tmp/` | 数据盘 | 200GB+ | ✅ |
| `/root/autodl-pub/` | 公共数据 | — | ✅ |

**所有大文件（模型、数据、输出）必须放 `/root/autodl-tmp/`**，否则关机丢失且系统盘会爆。

### 无卡模式 vs 有卡模式

| | 无卡 | 有卡 |
|------|:---:|:---:|
| 费用 | 极低或免费 | 按分钟计费 |
| 用途 | 传文件、装环境、下模型、写配置 | 训练/推理 |
| 网络 | 可能受限（下载易断） | 通常更稳 |

**推荐流程**：无卡模式做好一切准备工作 → 关机切有卡 → 直接训练 → 关机。

---

## 二、环境安装

### 确认 LLaMA-Factory 版本

```bash
python -c "import llamafactory; print(llamafactory.__version__)"
# 0.9.1.dev0 — 开发版，--help 不支持，但 train 正常
```

### CLI 入口差异

| 情况 | 命令 |
|------|------|
| 新版（正常安装） | `llamafactory-cli train ...` |
| 旧版/不完整安装 | `cd ~/LLaMA-Factory && python src/train_bash.py ...` |
| 模块找不到 | `pip install -e ".[torch,metrics]"` 重装 |

### 确认安装成功

```bash
which llamafactory-cli          # 应有输出
python -c "import llamafactory; print(llamafactory.__version__)"  # 应输出版本号
```

---

## 三、模型下载

### ModelScope（国内推荐，速度快但不稳定）

```bash
pip install modelscope -U
modelscope download --model qwen/Qwen2.5-7B-Instruct \
  --local_dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

注意：
- 大文件可能多次失败，反复重试最终能成功
- 支持断点续传，直接重跑命令即可

### HuggingFace 镜像（备选）

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface-hub -U
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False
```

### 后台下载（防断连）

```bash
nohup modelscope download --model qwen/Qwen2.5-7B-Instruct \
  --local_dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  > /root/autodl-tmp/download.log 2>&1 &
tail -f /root/autodl-tmp/download.log
```

### 验证下载完成

```bash
ls -lh /root/autodl-tmp/models/Qwen2.5-7B-Instruct/
# 应有 config.json, model-*.safetensors, tokenizer.json 等
```

---

## 四、Linux 操作速查

### 路径

| 命令 | 含义 |
|------|------|
| `pwd` | 当前路径 |
| `cd ~` | 回家目录（root 用户 = `/root`） |
| `cd ..` | 上一级 |
| `ls` 或 `ls .` | 当前目录内容 |
| `ls /root/data` | 查看指定目录 |

常见错误：已在 `/root/data` 下，`ls data` 会找 `/root/data/data`（不存在）。

### 编辑器

**优先装 nano**：
```bash
apt-get update && apt-get install -y nano
```

**vim 基本操作**：

| 操作 | 按键 |
|------|------|
| 进入编辑 | `i` |
| 退出编辑 | `Esc` |
| 保存退出 | `:wq` + 回车 |
| 强制退出（不保存） | `:q!` + 回车 |

vim 异常退出会留 `.swp` 文件，重新打开时先 `Q` 退出，再 `rm .xxx.swp` 删除。

### 磁盘检查

```bash
df -h                        # 查看所有挂载点
df -h /root/autodl-tmp/      # 数据盘空间
du -sh /root/* | sort -hr | head -10  # 哪些目录占空间
```

---

## 五、数据注册（dataset_info.json）

结构：
```json
{
  "chat_style_train": {
    "file_name": "/root/data/train.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"},
    "tags": {"role_tag": "from", "content_tag": "value",
             "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"}
  }
}
```

- `--dataset chat_style_train` → 对应 JSON 中的键名，不是文件名
- `system_tag: "system"` 不能写成 `columns.system`
- LLaMA-Factory v0.9.1+ 要求 conversations 中 system 后第一条必须是 human

---

## 六、训练命令关键点

```bash
llamafactory-cli train \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \  # 本地路径，不调 HF
  --dataset chat_style_train \
  --template qwen \                       # 必须
  --finetuning_type lora \
  --per_device_train_batch_size 1 \       # 7B 用 1，3B 用 2
  --bf16 \                                # 4090D 原生支持
  --quantization_bit 4 \                  # 7B 建议加
  ...
```

- 模型路径优先用本地预置 `/root/autodl-tmp/models/`，不用 HF 在线路径
- 外网不通就 `export HF_ENDPOINT=https://hf-mirror.com`
- bitsandbytes 如果 `pip install` 失败（AutoDL 无外网），去掉 `--quantization_bit 4`，7B LoRA 24GB 也能跑

---

## 七、踩坑清单

| # | 坑 | 解决 |
|:---:|------|------|
| 1 | 数据盘/系统盘搞混，模型下载到系统盘爆了 | 模型、数据、输出全放 `/root/autodl-tmp/` |
| 2 | ModelScope 下载大文件反复失败 | 多次重试，支持断点续传；或换 HF 镜像 |
| 3 | `llamafactory-cli` 找不到了 | `pip install -e .` 重装 |
| 4 | 训练只吃到 908 条（数据有 4592 条） | conversations 中 system 后第一条是 gpt 而非 human，LLaMA-Factory 丢弃了。本地 `build_lora_data.py` 已修复 |
| 5 | `--quantization_bit 4` 报错 bitsandbytes 缺失 | AutoDL 无外网，`pip install` 失败。去掉量化参数直接跑 |
| 6 | HuggingFace 直连不上 | 用本地模型路径或 `HF_ENDPOINT` 镜像 |
| 7 | vim 卡住退不出来 | `Esc` → `:q!` 强制退出；删 `.swp` 残留文件 |
| 8 | `No space left on device` | 系统盘满了，清 `/root` 下非必要文件，转存到 autodl-tmp |
| 9 | 终端输入中文后删除字符 → tokenizer 崩溃 | 删除操作产生异常空格/Unicode，破坏输入格式 | 用批量测试脚本代替交互式输入 |
| 10 | 打包 700MB 下载很慢 | output 目录包含 checkpoint、tokenizer 等冗余文件 | 只打 `adapter_model.safetensors` + `adapter_config.json`（~40MB） |
