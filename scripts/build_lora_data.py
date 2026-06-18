# -*- coding: utf-8 -*-
"""
LoRA 训练数据构建管道
Step 1: TXT emotion 修正 → Step 2: Turn 构建 → Step 3: 话题分割 → Step 4: ShareGPT 输出
"""

import json, sys, io, re
from pathlib import Path
from datetime import timedelta
from collections import Counter

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(r"E:\狸喵galgame")
SESSION_DIR = ROOT / "sessions"
OUTPUT_DIR = ROOT / "lora_data"

TIME_GAP_MINUTES = 30      # 话题分割时间阈值
CONTEXT_TURNS = 10         # human 上下文包含的最近 turns
TRAIN_RATIO = 0.8          # 训练集比例


# ========= Step 1: TXT emotion 修正 =========

def load_txt_emotions():
    """从各文件夹的 TXT 文件加载黄金标注 emotion"""
    txt_data = {}
    for txt_path in sorted(ROOT.glob("*/*.txt")):
        folder = txt_path.parent.name
        if folder in ("venv", "splitted", "results", "sessions", "strips", "lora_data"):
            continue
        try:
            msgs = json.loads(txt_path.read_text("utf-8"))
            txt_data[(folder, txt_path.stem)] = [m["meta"]["emotion"] for m in msgs]
        except Exception:
            pass
    return txt_data


def step1_apply(txt_emotions, sessions):
    """TXT emotion → sessions，其余去除 emotion"""
    fixed, stripped = 0, 0
    for sess in sessions:
        key = (sess["category"], Path(sess["source_image"]).stem)
        if key in txt_emotions:
            gold = txt_emotions[key]
            msgs = sess.get("messages", [])
            for i in range(min(len(msgs), len(gold))):
                if "meta" not in msgs[i]:
                    msgs[i]["meta"] = {}
                msgs[i]["meta"]["emotion"] = gold[i]
            fixed += 1
        else:
            for m in sess.get("messages", []):
                if "meta" in m and "emotion" in m["meta"]:
                    del m["meta"]["emotion"]
                if "meta" in m and not m["meta"]:
                    del m["meta"]
            stripped += 1
    print(f"Step 1: {fixed} 修正, {stripped} 去除 emotion")
    return sessions


# ========= Step 2: Turn Construction =========

def step2_build_turns(messages):
    """同 sender 连续消息合并为一个 turn"""
    turns = []
    for msg in messages:
        s = msg.get("sender", "me")
        t = msg.get("text", "")
        ts = msg.get("time", "")
        emo = msg.get("meta", {}).get("emotion", "")

        if turns and turns[-1]["sender"] == s:
            # 追加文本
            turns[-1]["text"] += "\n" + t
            if ts:
                turns[-1]["time"] = ts
            if emo:
                turns[-1]["_emos"].append(emo)
        else:
            turns.append({"sender": s, "text": t, "time": ts, "_emos": [emo] if emo else []})

    # 计算主导 emotion
    for turn in turns:
        if turn["_emos"]:
            turn["emotion"] = Counter(turn["_emos"]).most_common(1)[0][0]
        else:
            turn["emotion"] = None
        del turn["_emos"]
    return turns


# ========= Step 3: 话题分割 =========

def parse_time(ts):
    if not ts:
        return None
    try:
        h, m = map(int, ts.split(":"))
        return timedelta(hours=h, minutes=m)
    except Exception:
        return None


def step3_split_topics(turns):
    """按时间间隙分割话题块"""
    topics = []
    block = []
    last_abs = 0   # 绝对分钟计数器

    for turn in turns:
        t = parse_time(turn.get("time"))
        current = t.total_seconds() / 60 if t else last_abs + 1

        # 检测时间回绕或大间隔
        if block:
            diff = current - last_abs
            if diff < 0:
                diff += 24 * 60   # 跨天
            if diff > TIME_GAP_MINUTES:
                topics.append(block)
                block = []

        block.append(turn)
        last_abs = current

    if block:
        topics.append(block)
    return topics


# ========= Step 4: ShareGPT 输出 =========

def step4_export(all_data):
    samples = []
    for sess, turns in all_data:
        source_tag = f"{sess['category']}/{sess['source_image']}"
        topics = step3_split_topics(turns)

        for block in topics:
            # 找到第一个 me turn（对话必须以 human 开头）
            first_me = None
            for i, t in enumerate(block):
                if t["sender"] == "me":
                    first_me = i
                    break
            if first_me is None:
                continue  # 整个块没有 me，跳过

            # 为第一个 me 之后的 them turn 构造训练样本
            for i, turn in enumerate(block):
                if turn["sender"] != "them":
                    continue
                if i <= first_me:
                    continue  # 跳过第一个 me 之前的 them

                # 上下文起始位置，确保第一个是 human（me）
                ctx_start = max(first_me, i - CONTEXT_TURNS)
                while ctx_start < i and block[ctx_start]["sender"] == "them":
                    ctx_start += 1
                if ctx_start >= i:
                    continue  # 前面全是 them，无法构成训练样本

                ctx = block[ctx_start:i]

                convs = []

                # system prompt
                emo = turn.get("emotion")
                emo_map = {
                    "teasing": "以调侃、打趣的语气聊天",
                    "sarcastic": "以反讽、阴阳怪气的语气聊天",
                    "affectionate": "以亲昵、撒娇的语气聊天",
                }
                inst = emo_map.get(emo)
                sys_msg = "你正在微信上与朋友聊天。"
                if inst:
                    sys_msg += f" {inst}"
                convs.append({"from": "system", "value": sys_msg})

                # 上下文（从 first_me 开始 → 第一个一定是 human）
                for ct in ctx:
                    role = "gpt" if ct["sender"] == "them" else "human"
                    convs.append({"from": role, "value": ct["text"]})

                # 目标 (them turn)
                convs.append({"from": "gpt", "value": turn["text"]})

                samples.append({"source": source_tag, "conversations": convs})

    # 按文件名日期排序
    def get_date(s):
        m = re.search(r"(\d{8})", s["source"])
        return int(m.group(1)) if m else 20220101

    samples.sort(key=get_date)
    split = int(len(samples) * TRAIN_RATIO)

    return samples[:split], samples[split:]


# ========= 主流程 =========

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 加载所有 sessions
    sessions = []
    for sp in sorted(SESSION_DIR.glob("*.json")):
        sessions.append(json.loads(sp.read_text("utf-8")))
    print(f"加载 {len(sessions)} 个 session\n")

    # Step 1
    txt_emotions = load_txt_emotions()
    print(f"找到 {len(txt_emotions)} 个 TXT 黄金标注")
    step1_apply(txt_emotions, sessions)

    # Step 2: 每个 session 的消息 → turns
    all_data = []
    total_turns = 0
    for sess in sessions:
        msgs = sess.get("messages", [])
        if not msgs:
            continue
        turns = step2_build_turns(msgs)
        all_data.append((sess, turns))
        total_turns += len(turns)
    print(f"Step 2: {total_turns} 个 turns")

    # Step 3 + 4: 分割 + 输出
    train, valid = step4_export(all_data)
    print(f"Step 3+4: {len(train) + len(valid)} 个训练示例")

    with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "valid.json", "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)

    stats = {
        "sessions": len(sessions),
        "total_turns": total_turns,
        "train_samples": len(train),
        "valid_samples": len(valid),
        "emotion_distribution": dict(Counter(
            s["conversations"][0]["value"].split("。")[-1].replace(" ","")
            for s in train
        )),
        "avg_conversation_length": sum(
            len(s["conversations"]) for s in train + valid
        ) / max(len(train) + len(valid), 1),
    }
    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成 ===")
    print(f"  lora_data/train.json : {len(train)} 训练示例")
    print(f"  lora_data/valid.json : {len(valid)} 验证示例")
    print(f"  lora_data/stats.json : 统计")


if __name__ == "__main__":
    main()
