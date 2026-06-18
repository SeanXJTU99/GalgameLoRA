# -*- coding: utf-8 -*-
"""sessions/ → ShareGPT 训练数据（按日期合并、无 system、不合并同 sender、human 开头）"""
import json, sys, io, re
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(r"E:\狸喵galgame")
SESSION_DIR = ROOT / "sessions"
OUTPUT_DIR = ROOT / "lora_data_v2"
TRAIN_RATIO = 0.8
TEST_MODE = False
TEST_DATES = 3
MAX_CTX = 30  # 每样本最多上下文消息数

# 1. 读取所有 session，按日期分组
date_groups = defaultdict(list)
for sp in sorted(SESSION_DIR.glob("*.json")):
    # 从文件名提日期: 0107道士下山_20220107_122955.json → 20220107, 122955
    m = re.search(r"(\d{8})_(\d{6})", sp.name)
    if not m:
        continue
    date, time = m.group(1), m.group(2)
    date_groups[date].append((time, sp))

dates = sorted(date_groups.keys())
print(f"共 {len(dates)} 个日期, {sum(len(v) for v in date_groups.values())} 个 session")

if TEST_MODE:
    dates = dates[:TEST_DATES]

# 2. 按日期生成样本
samples = []
total_skipped = 0

for date in dates:
    # 按时序排列同日 session
    sessions = [sp for _, sp in sorted(date_groups[date])]
    # 拼接同日所有消息
    convs = []
    for sp in sessions:
        sess = json.loads(sp.read_text("utf-8"))
        for msg in sess.get("messages", []):
            role = "human" if msg.get("sender") == "me" else "gpt"
            text = (msg.get("text") or "").strip()
            if text:
                convs.append({"from": role, "value": text})

    # 找第一个 human
    first_h = None
    for i, m in enumerate(convs):
        if m["from"] == "human":
            first_h = i
            break
    if first_h is None:
        total_skipped += len(convs)
        continue

    # 每个 gpt → 一条样本（滑动窗口，最多 MAX_CTX 条上下文）
    for i in range(len(convs)):
        if convs[i]["from"] != "gpt":
            continue
        # 上下文起点：不早于 first_h，最多取 MAX_CTX 条
        start = max(first_h, i - MAX_CTX)
        while start < i and convs[start]["from"] != "human":
            start += 1
        if start >= i:
            total_skipped += 1
            continue
        ctx = [{"from": "system", "value": "你正在微信上与朋友聊天。"}]
        ctx.extend(convs[start : i+1])
        samples.append({"conversations": ctx})

    # 统计浪费：first_h 之前的 gpt + 窗口中无法凑出 human 开头的
    total_skipped += sum(1 for m in convs[:first_h] if m["from"] == "gpt")

# 3. 按日期切分 train/valid
split = int(len(samples) * TRAIN_RATIO)
train, valid = samples[:split], samples[split:]

if TEST_MODE:
    print(f"\n=== 测试：前 {TEST_DATES} 个日期 ===\n")
    for i, s in enumerate(samples[:5]):
        print(f"--- 样本 {i+1} ({len(s['conversations'])} 条) ---")
        for m in s["conversations"]:
            print(f"  [{m['from']}] {m['value'][:80]}")
        print()
    print(f"样本: {len(samples)}, 跳过 gpt: {total_skipped} ({total_skipped/max(total_skipped+len(samples),1)*100:.1f}%)")
else:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "valid.json", "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    print(f"train: {len(train)}, valid: {len(valid)}, 跳过 gpt: {total_skipped}")
    print(f"→ {OUTPUT_DIR}/")
