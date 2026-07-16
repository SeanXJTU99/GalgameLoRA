# -*- coding: utf-8 -*-
"""对话后异步抽取记忆（编排 LLM，server.py BackgroundTasks 调用）"""
import json

from openai import OpenAI

import config
from .models import Memory
from .store import add_memory

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


EXTRACT_PROMPT = """从以下对话中提取关于用户"喵喵"的新事实或偏好，每条 ≤20 字。
没有新信息则返回空数组。只输出 JSON 数组，不要任何其他内容。
importance 规则：强烈情绪 0.7-0.9，偏好陈述 0.6，普通事实 0.5。
格式：[{{"content": "...", "importance": 0.5}}]

用户: {user_msg}
AI: {ai_reply}"""


def extract_memories(user_msg: str, ai_reply: str) -> list[Memory]:
    resp = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user",
                   "content": EXTRACT_PROMPT.format(user_msg=user_msg, ai_reply=ai_reply)}],
        temperature=0.1,
        max_tokens=256,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):  # 剥代码围栏
        text = text.strip("`").lstrip("json").strip()
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [Memory(content=it["content"], importance=float(it.get("importance", 0.5)))
            for it in items if isinstance(it, dict) and it.get("content")]


def extract_and_save(user_id: str, user_msg: str, ai_reply: str) -> list[str]:
    """抽取 + 入库（异步后台任务入口），抽取失败静默跳过不影响对话"""
    try:
        memories = extract_memories(user_msg, ai_reply)
    except Exception:
        return []
    return [add_memory(user_id, m) for m in memories]
