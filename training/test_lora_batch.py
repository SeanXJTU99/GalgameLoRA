# AutoDL 上执行，批量测试训练好的 LoRA
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/output/chat_style_lora"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, LORA_PATH)

SYSTEM_PROMPT = "你是阿狸，一个爱撒娇、喜欢黏着喵喵的女生。你正在和男朋友喵喵微信聊天。回复短小精炼，爱撒娇。"

def chat(user_input):
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=128, temperature=0.8, do_sample=True,
        pad_token_id=tokenizer.eos_token_id, top_p=0.9,
    )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()

tests = [
    "小河狸乖乖，把门儿开开",
    "跟我说说话嘛，喵喵想你了",
    "今天天气真好，出去玩吗",
    "你怎么不回我消息",
    "我饿了，想吃你做的饭",
    "嘻嘻你承认你是我家的了？",
]

print("=" * 50)
for t in tests:
    print(f"你: {t}")
    print(f"AI: {chat(t)}")
    print()
print("=" * 50)
