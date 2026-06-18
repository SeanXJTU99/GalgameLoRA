# -*- coding: utf-8 -*-
"""sessions/ → pure_conversations.json，纯对话，无多余字段"""
import json, sys, io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(r"E:\狸喵galgame")
SESSION_DIR = ROOT / "sessions"
OUTPUT = ROOT / "pure_conversations.json"

ROLE_MAP = {"me": "human", "them": "gpt"}

result = []
for sp in sorted(SESSION_DIR.glob("*.json")):
    sess = json.loads(sp.read_text("utf-8"))
    convs = []
    for msg in sess.get("messages", []):
        role = ROLE_MAP.get(msg.get("sender"))
        text = (msg.get("text") or "").strip()
        if not role or not text:
            continue
        convs.append({"from": role, "value": text})
    if convs:
        result.append({"conversations": convs})

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

total_msgs = sum(len(r["conversations"]) for r in result)
print(f"导出 {len(result)} 个会话, {total_msgs} 条消息 → {OUTPUT}")
