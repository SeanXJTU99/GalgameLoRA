# LoRA 微调实施指南

> 最后更新：2026-06-17 | 基座：Qwen2.5-3B / 7B | 平台：AutoDL 4090D | 框架：LLaMA-Factory

---

## 前置准备

### 你需要上传到云的文件

| 文件 | 说明 |
|------|------|
| `lora_data/train.json` | 训练集（~4,592 条） |
| `lora_data/valid.json` | 验证集（~1,148 条） |
| `lora_data/dataset_info.json` | 数据集注册配置 |

这三个文件通过 `build_lora_data.py` 生成，在本地 `E:\狸喵galgame\lora_data\` 下。

> **注意**：数据管道已过滤掉 system 后第一个对话是 `gpt`（不是 `human`）的无效样本。否则 LLaMA-Factory v0.9+ 会丢弃这些样本，导致实际训练量远小于预期。

### 不需要的东西

- **API Key**：训练全程零 API 调用
- **模型文件**：镜像通常已预置，或通过 HuggingFace 镜像自动下载

---

## 一、租用 GPU（AutoDL）

1. 打开 [AutoDL](https://www.autodl.com)，注册/登录
2. 充值 ¥10-20（支付宝/微信）
3. 进入「算力市场」
4. 筛选：**GPU 类型 → RTX 4090D**（24GB 显存）
5. 镜像选择：**社区镜像**，搜索 `LLaMA-Factory`，选最新版本（v0.9+）
6. 按量计费，点击「创建实例」
7. 等待分配（通常 1-2 分钟）
8. 点击「JupyterLab」进入

> **计费规则**：开机即按分钟扣费，关机停费。训完立刻关机，不要挂着。

---

## 二、上传数据

### 方法 1：JupyterLab 拖拽（推荐）

1. 左侧文件面板导航到 `/root/`
2. 新建文件夹：右键 → New Folder → 命名 `data`
3. 进入 `/root/data/`
4. 把本地的 `train.json`、`valid.json`、`dataset_info.json` 三个文件直接拖入左侧面板

### 方法 2：AutoDL 文件管理

1. 在实例列表点击「更多」→「文件管理」
2. 导航到 `/root/data/`
3. 点击上传，选择三个文件

### 验证上传

```bash
ls -lh /root/data/
# 应该看到 train.json、valid.json、dataset_info.json
python -c "import json; d=json.load(open('/root/data/train.json')); print(len(d))"
# 应该输出 4592
```

---

## 三、注册数据集

### 1. 找到 LLaMA-Factory

```bash
ls ~/LLaMA-Factory/data/dataset_info.json
```

如果找不到，搜索：

```bash
find ~ -name "dataset_info.json" -type f 2>/dev/null
```

### 2. 编辑 dataset_info.json

```bash
nano ~/LLaMA-Factory/data/dataset_info.json
```

在文件末尾的 `}` **之前**，添加以下内容（注意逗号）：

```json
"chat_style_train": {
  "file_name": "/root/data/train.json",
  "formatting": "sharegpt",
  "columns": {"messages": "conversations"},
  "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"}
},
"chat_style_valid": {
  "file_name": "/root/data/valid.json",
  "formatting": "sharegpt",
  "columns": {"messages": "conversations"},
  "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"}
}
```

**关键**：
- 如果 `chat_style_train` 前面有上一项，需要在上一项的 `}` 后加逗号
- `system` 是 conversations 内部的角色（`from: system`），**不是**顶层字段，所以用 `tags.system_tag`，不能用 `columns.system`

保存退出：`Ctrl+O` → Enter → `Ctrl+X`

---

## 四、找模型路径

### 检查镜像是否已预置

```bash
find /root/autodl-tmp -path "*Qwen*" -type d 2>/dev/null | head -5
ls ~/.cache/huggingface/hub/ | grep Qwen 2>/dev/null
```

### 模型路径选择

| 情况 | `--model_name_or_path` |
|------|------|
| 镜像已预置 | `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` 或类似路径 |
| 无预置，需下载 | `Qwen/Qwen2.5-7B-Instruct`（HuggingFace 自动拉取） |

### 无 VPN 下载

**方案 A**：HuggingFace 镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**方案 B**：ModelScope 国内源（推荐，速度更快）
```bash
pip install modelscope -q
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models')"
```
下载后模型路径：`/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct`

---

## 五、训练

### 5.1 确认入口

```bash
which llamafactory-cli
```

有输出 → 用 `llamafactory-cli train`。无输出 → 用 `python src/train_bash.py`。

### 5.2 选模型：3B vs 7B

| | 3B（快速验证） | 7B（正式训练） |
|------|:---:|:---:|
| 下载量 | ~6GB | ~14GB |
| 显存占用 | ~10GB (FP16) | ~8GB (QLoRA 4bit) |
| 训练速度 | 快 | 较慢 |
| 量化 | 不需要 | **必须** `--quantization_bit 4` |
| Batch size | 2 | 1 |
| 总步数 | 574 | 1148 |
| 预计时间 | ~40min | ~1.5h |

### 5.3 3B 训练命令

```bash
cd ~/LLaMA-Factory
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset chat_style_train \
  --eval_dataset chat_style_valid \
  --output_dir /root/output/chat_style_lora \
  --stage sft \
  --do_train \
  --do_eval \
  --finetuning_type lora \
  --template qwen \
  --lora_rank 8 \
  --lora_target all \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --cutoff_len 1024 \
  --logging_steps 50 \
  --save_steps 500 \
  --eval_steps 500 \
  --bf16 \
  --overwrite_output_dir
```

### 5.4 7B 训练命令

```bash
# 先安装量化依赖
pip install bitsandbytes -q

cd ~/LLaMA-Factory
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset chat_style_train \
  --output_dir /root/output/chat_style_lora \
  --stage sft \
  --do_train \
  --finetuning_type lora \
  --template qwen \
  --lora_rank 8 \
  --lora_target all \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --cutoff_len 1024 \
  --logging_steps 50 \
  --save_steps 500 \
  --bf16 \
  --quantization_bit 4 \
  --overwrite_output_dir
```

> 7B 去掉了 `--eval_dataset` 和 `--do_eval`：QLoRA 加 eval 可能显存不够。先训练，本地再测效果。

### 参数说明

| 参数 | 3B | 7B | 原因 |
|------|:---:|:---:|------|
| `--template qwen` | ✅ | ✅ | **必须**，Qwen 家族专用 |
| `--lora_rank 8` | ✅ | ✅ | 5K 样本学风格，不调 16 |
| `--num_train_epochs 2` | ✅ | ✅ | 不调 3，防过拟合 |
| `--bf16` | ✅ | ✅ | 4090D 原生支持 |
| `--quantization_bit 4` | ❌ | ✅ | 7B 太大，必须 4bit |
| `--per_device_train_batch_size` | 2 | 1 | 7B 显存更紧张 |

### 训练前快速验证（推荐）

先跑 1 epoch 干燥验证，确认数据正确解析：

```bash
llamafactory-cli train \
  --model_name_or_path <模型路径> \
  --dataset chat_style_train \
  --output_dir /root/output/test \
  --stage sft \
  --do_train \
  --finetuning_type lora \
  --template qwen \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --max_steps 10 \
  --cutoff_len 1024 \
  --bf16 \
  --overwrite_output_dir
```

如果 `Num examples` 显示 ≥ 4000，说明数据正确。只有几百条说明数据有问题，检查 `dataset_info.json` 和 `train.json`。

---

## 六、训练完成后

### 6.1 查看 loss

关注：
- Train loss 持续下降 = 正常
- Eval loss 不反弹 = 不过拟合
- Loss 震荡不降 = 有问题，中断

### 6.2 打包下载

**只需下载 LoRA 权重（~40MB）**，不需要整个 output 目录（700MB）：

```bash
cd /root/autodl-tmp/output/chat_style_lora
tar czf /root/autodl-tmp/lora_weights_only.tar.gz adapter_model.safetensors adapter_config.json
```

推理只需这两个文件 + 基座模型。tokenizer 等文件基座模型自带。

### 6.3 本地推理测试

推理时用**人格化 system prompt**（训练时用泛用 prompt，推理时加身份）：

```python
SYSTEM_PROMPT = "你是阿狸，...你正在和男朋友喵喵微信聊天。回复短小精炼，爱撒娇。"

# Qwen2 聊天格式
text = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{user_input}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

model = PeftModel.from_pretrained(base_model, lora_path)
```

**注意**：AutoDL 终端输入中文时删除字符会导致 Unicode 损坏 → tokenizer 崩溃。使用批量脚本（`test_lora_batch.py`）代替交互式输入。

### 6.4 关机

**下载完成后立刻关机**。AutoDL 控制台 → 实例 → 关机。

---

## 七、Colab vs AutoDL 对比

| | Colab 免费 T4 | AutoDL 4090D |
|------|:---:|:---:|
| 显存 | 16GB | 24GB |
| 精度 | FP16 | BF16 |
| 速度 | ~6.4s/step | ~4-5s/step |
| 3B 时间 | ~1h | ~40min |
| 7B 时间 | 跑不了 | ~1.5h |
| 费用 | ¥0 | ≈¥2-5 |
| 断连风险 | 高 | 低 |
| 数据持久 | 断开即清空 | 关机保留磁盘 |

---

## 八、故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| "Dropped invalid example" 警告，实际训练量远小于数据量 | train.json 中 conversations 的 system 后第一条不是 human | 重新运行本地 `build_lora_data.py` 生成数据（已修复），再上传覆盖 |
| `bitsandbytes` 找不到 | `--quantization_bit 4` 需要 bitsandbytes | `pip install bitsandbytes -q` |
| bitsandbytes 安装后崩溃 | 某些环境下 bitsandbytes Gaudi 检测有问题 | 去掉 `--quantization_bit 4`，7B 在 24GB 上可能刚好够 |
| `llamafactory-cli: command not found` | 旧版 LLaMA-Factory | 改用 `python src/train_bash.py` |
| `No dataset named 'chat_style_train'` | 未注册或 JSON 格式错误 | 检查 `dataset_info.json` 逗号、括号 |
| 下载模型慢/失败 | 无 VPN | `export HF_ENDPOINT=https://hf-mirror.com` |
| system 角色报错 | 用了 `columns.system` | 改用 `tags.system_tag: "system"` |
| OOM | 显存不足 | 降 batch_size 或加 `--quantization_bit 4` |
| `warmup_ratio` 警告 | Transformers 弃用 | 功能仍可用，忽略 |
| 模型输出 `[sticker:xxx]` | 训练数据自带 sticker 占位符 | 正常 |
| 模型分不清 me/them | system prompt 太泛 | 推理时加身份指令 |
| `pip install` 连不上外网 | AutoDL 无外网 | 去掉 `--quantization_bit 4`（不装 bitsandbytes 也能跑 7B LoRA） |
| HuggingFace 连不上 | AutoDL 无外网 | 用镜像已预置的本地模型路径（`/root/autodl-tmp/models/`）或 `export HF_ENDPOINT=https://hf-mirror.com` |

---

## 九、AutoDL 注意事项

### 外网不通
- AutoDL 默认无外网，`pip install` 和 `huggingface.co` 直连大概率失败
- **不要用 `--quantization_bit 4`**，bitandbytes 无法在线安装。7B LoRA BF16 在 24GB 上刚好够
- 如果 bitsandbytes 确实需要，试试 `pip install bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 模型路径
- **优先用镜像预置**：`ls /root/autodl-tmp/models/` 查看
- 如果本地无缓存，设置镜像后让 HF 自动下载

### 社区镜像
- 搜索 `LLaMA-Factory` 结果很多，选版本号最高的（v0.9.1+）
- 进实例后 `which llamafactory-cli` 确认是新版

---

## 十、训练历史

| 版本 | system prompt | 模型 | 结果 |
|------|:---|:---|------|
| v1 | `你正在微信上与朋友聊天。`（12 字） | 3B | ✅ loss 5.57→3.21 |
| v2 | `你是阿狸...`（120 字完整人格） | 3B | ❌ 退化：输出 sticker、角色混乱 |
| v3 | 同 v1 | 7B | ✅ loss 3.39，37min，效果≈v1 |

**7B vs 3B 结论**：无显著差异。风格迁移瓶颈在数据质量和 system prompt 策略，不在模型大小。3B 足够。

**当前最优方案**：训练用泛用短 prompt + 推理用带身份的长 prompt。

**v2 失败根因**：
- 120 字 prompt 重复 4592 次 → 模型过拟合到背诵指令
- `cutoff_len=1024` 下吃掉 ~60 tokens，对话上下文不足
- "少用 emoji" 指令与训练数据中 `[sticker:xxx]` 直接矛盾

**教训**：训练时 system prompt 应极简（≤20 字），风格从数据自然学习。身份指令留在推理阶段使用。
