# -*- coding: utf-8 -*-
"""Layer 0/3 风格基线 — 同 20 prompt 跑两个 llama-server 端点（v4 vs 基座），
产出并排 JSONL 供人工/GPT 盲评。

用法：
  python eval/style_baseline.py --v4-url http://localhost:8080 \
      --base-url http://localhost:8081 --out baseline.jsonl
  只测单模型可省略 --base-url。

双路线风格稀释实验（P2.6）：
  CONTEXT_MODE=minimal python eval/style_baseline.py ... --out minimal.jsonl
  CONTEXT_MODE=rich    python eval/style_baseline.py ... --out rich.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # agent/ 根

import config  # noqa: E402
from generation import llama_client  # noqa: E402

# 取自 低显存无损推理.md 已验证 prompt + 常见陪伴场景
PROMPTS = [
    "想你了",
    "老婆怎么这么主动？",
    "抱抱啦老婆，一整天想死你了",
    "今晚一起去山姆吧买点好吃的",
    "阿狸在家想我了没有",
    "小河狸乖乖，把门儿开开",
    "今天好累啊",
    "我升职啦！",
    "睡不着，陪我聊聊",
    "你猜我今天遇到谁了",
    "晚饭吃什么好呢",
    "我好像感冒了",
    "周末想出去玩，去哪好",
    "刚才开会被老板骂了",
    "给你带了小蛋糕哦",
    "凡凡最近怎么不理我",
    "下雨了，没带伞",
    "我妈又催婚了，烦死",
    "新买的键盘到了，超好看",
    "晚安啦宝贝",
]


def run_endpoint(url: str, api_key: str = "") -> list[str]:
    config.LLAMA_SERVER_URL = url
    config.LLAMA_API_KEY = api_key
    replies = []
    for i, p in enumerate(PROMPTS, 1):
        reply = llama_client.generate(config.PERSONA, p)
        replies.append(reply)
        print(f"[{i}/{len(PROMPTS)}] {p} → {reply[:40]}")
    return replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-url", required=True, help="v4 LoRA merged llama-server 地址")
    ap.add_argument("--base-url", default="", help="基座 llama-server 地址（可选）")
    ap.add_argument("--api-key", default="", help="llama-server --api-key")
    ap.add_argument("--out", default="baseline.jsonl")
    args = ap.parse_args()

    print("=== v4 LoRA ===")
    v4 = run_endpoint(args.v4_url, args.api_key)
    base = None
    if args.base_url:
        print("=== 基座 ===")
        base = run_endpoint(args.base_url, args.api_key)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, p in enumerate(PROMPTS):
            row = {"prompt": p, "v4": v4[i]}
            if base:
                row["base"] = base[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"→ {args.out}（{len(PROMPTS)} 条，供盲评）")


if __name__ == "__main__":
    main()
