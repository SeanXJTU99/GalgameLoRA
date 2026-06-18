# 在 AutoDL 终端执行，测试训练好的 LoRA 效果
# 前提：实例已开机，模型和数据都在 /root/autodl-tmp/ 下

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/output/chat_style_lora"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, LORA_PATH)

SYSTEM_PROMPT = "你是阿狸，一个爱撒娇、喜欢黏着喵喵的女生。你正在和男朋友喵喵微信聊天。回复短小精炼，爱撒娇。"

def chat(user_input):
    user_input = user_input.strip()
    if not user_input:
        return ""
    # Qwen2 聊天格式
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    try:
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=128, temperature=0.8, do_sample=True,
            pad_token_id=tokenizer.eos_token_id, top_p=0.9,
        )
        response = tokenizer.decode(
            outputs[0][len(inputs[0]):], skip_special_tokens=True
        )
        return response.strip()
    except Exception as e:
        return f"[错误: {e}]"

print("=" * 50)
print("输入 quit 退出")
print("=" * 50)

while True:
    user = input("你: ")
    if user.lower() == "quit":
        break
    resp = chat(user)
    print(f"AI: {resp}")
    print()
