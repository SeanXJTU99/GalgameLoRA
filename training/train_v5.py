# -*- coding: utf-8 -*-
"""v5 LoRA: attention-only, top 6 layers, rank=8 — 纯风格迁移，用 PEFT + transformers
用法：python training/train_v5.py   (需要 GPU ≥ 16G)
"""
import json, sys, io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset

ROOT = Path(__file__).parent.parent

# ── 加载数据 ──
with open(ROOT / "data" / "train.json", encoding="utf-8") as f:
    raw = json.load(f)

def format_sample(conv):
    """ShareGPT → Qwen chat template 文本"""
    texts = []
    for m in conv["conversations"]:
        role = "user" if m["from"] == "human" else ("assistant" if m["from"] == "gpt" else "system")
        texts.append({"role": role, "content": m["value"]})
    return tokenizer.apply_chat_template(texts, tokenize=False, add_generation_prompt=False)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

formatted = [format_sample(s) for s in raw]
dataset = Dataset.from_dict({"text": formatted})

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# ── 加载模型 ──
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# ── v5 LoRA: attention-only, top 6 layers, rank=8 ──
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        # 顶部 6 层 (22-27)，只 attention，不改 FFN
        r"model\.layers\.2[2-7]\.self_attn\.q_proj",
        r"model\.layers\.2[2-7]\.self_attn\.k_proj",
        r"model\.layers\.2[2-7]\.self_attn\.v_proj",
        r"model\.layers\.2[2-7]\.self_attn\.o_proj",
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 训练 ──
training_args = TrainingArguments(
    output_dir=str(ROOT / "weights" / "v5"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    overwrite_output_dir=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained(str(ROOT / "weights" / "v5"))
print("→ v5 weights saved to weights/v5/")
