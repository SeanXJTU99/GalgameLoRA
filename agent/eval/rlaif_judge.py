# -*- coding: utf-8 -*-
"""DeepSeek 3维裁判 → 过滤偏好对 → DPO 格式 JSON

用法：
  python eval/rlaif_judge.py --in rlaif_candidates.jsonl --out ../data/dpo_train.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from openai import OpenAI

JUDGE_PROMPT = """你正在评估聊天AI"阿狸"的回复质量。阿狸是个爱撒娇、黏人的女生，在和男朋友喵喵微信聊天。

用户喵喵说："{user_msg}"

回复A：{reply_a}

回复B：{reply_b}

请对每条回复在以下维度打分（1-5分整数），并判断哪条更好。
- style（风格一致性）：是否像真实微信聊天（简短口语、有语气词/颜文字/sticker占位符）
- emotion（情感真实感）：是否有女友撒娇感，亲切自然不敷衍
- length（回复粒度）：是否简短精炼（阿狸很少发长回复，1-2句为主）

只输出JSON，不要任何其他内容：
{{"A":{{"style":3,"emotion":2,"length":5}}, "B":{{"style":4,"emotion":4,"length":4}}, "better":"B", "total_diff":5}}"""


def score_pair(prompt: str, reply_a: str, reply_b: str) -> dict | None:
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            user_msg=prompt, reply_a=reply_a, reply_b=reply_b)}],
        temperature=0.1,
        max_tokens=200,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 正则兜底
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-diff", type=int, default=2,
                    help="总分差 ≥ N 才收入偏好对")
    args = ap.parse_args()

    pairs = []
    judged = 0
    with open(args.infile, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            judged += 1
            result = None
            for attempt in range(3):
                try:
                    result = score_pair(row["prompt"], row["reply_a"], row["reply_b"])
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        print(f"  skip [{judged}] {row['prompt'][:20]}: {e}")
            if result is None:
                continue
            diff = int(result.get("total_diff", 0))
            if diff < args.min_diff:
                continue
            better = result["better"]
            chosen = row["reply_a"] if better == "A" else row["reply_b"]
            rejected = row["reply_b"] if better == "A" else row["reply_a"]
            pairs.append({
                "conversations": row["conversations"],
                "chosen": chosen,
                "rejected": rejected,
            })
            if judged % 50 == 0:
                print(f"[{judged}] 已判定 {len(pairs)} 对收入")

            time.sleep(0.5)  # 避免 API 限流

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"→ {args.out}（{len(pairs)} 对 / {judged} 条候选）")


if __name__ == "__main__":
    main()
