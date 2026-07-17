# -*- coding: utf-8 -*-
"""train.json 每条 prompt → v4 LoRA 生成两条回复（temp 0.7 / 1.0），产出偏好候选 JSONL

用法：
  python eval/rlaif_generate.py --url https://u1055448-8174-e8fb8c7b.westc.seetacloud.com:8443 \
      --api-key ali2026sk --data ../data/train.json --out rlaif_candidates.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from generation import llama_client  # noqa: E402


def load_prompts(data_path: str, limit: int = 0) -> list[dict]:
    """读 ShareGPT train.json → 提取 [(conversations, last_human_msg), ...]"""
    with open(data_path, encoding="utf-8") as f:
        samples = json.load(f)
    if limit:
        samples = samples[:limit]
    return samples


def extract_prompt(sample: dict) -> str:
    """从 ShareGPT conversations 取最后一条 human 消息"""
    convs = sample["conversations"]
    # 最后一条是 gpt（训练目标），倒数第二条是 human（用户消息）
    human_msgs = [c["value"] for c in convs if c["from"] == "human"]
    if not human_msgs:
        return None
    return human_msgs[-1]  # 取最后一条 human，忽略前面的上下文


def generate_two(sample: dict, url: str, api_key: str) -> dict:
    """一条 prompt 生成两次（不同 temperature），返回 {"prompt": ..., "conversations": ..., "reply_a": ..., "reply_b": ...}"""
    config.LLAMA_SERVER_URL = url
    config.LLAMA_API_KEY = api_key
    prompt = extract_prompt(sample)
    if prompt is None:
        return None
    a = llama_client.generate(config.PERSONA, prompt, temperature=0.7)
    b = llama_client.generate(config.PERSONA, prompt, temperature=1.0)
    return {"conversations": sample["conversations"], "prompt": prompt, "reply_a": a, "reply_b": b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="rlaif_candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    args = ap.parse_args()

    samples = load_prompts(args.data, args.limit)
    print(f"{len(samples)} 条 prompt，每条生成 2 次…")

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples, 1):
            row = generate_two(s, args.url, args.api_key)
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            count += 1
            if i % 50 == 0:
                print(f"[{i}/{len(samples)}] {row['prompt'][:30]} → {row['reply_a'][:20]} | {row['reply_b'][:20]}")
            time.sleep(0.1)  # 不压垮云端 CPU

    print(f"→ {args.out}（{count} 条）")


if __name__ == "__main__":
    main()
