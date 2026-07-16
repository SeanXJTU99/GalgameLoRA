# -*- coding: utf-8 -*-
"""上下文组装 — 风格稀释双路线，CONTEXT_MODE 切换（策略开关，非 git 分支）

rich    : 编排 LLM 组装 ≤200 字富上下文（记忆+情境+情绪），每轮 1 次 API 调用
minimal : 贴近训练分布，人格 + 记忆摘要 ≤30 字，零 API 调用

v2 教训：120 字 system prompt 即导致训练退化 → rich 路线是否可行需实验验证，
未验证前默认 minimal（config.CONTEXT_MODE）。
返回：{"system": str, "emotion_label": str}
"""
import json

from openai import OpenAI

import config

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


ASSEMBLE_PROMPT = """你是"阿狸"聊天 agent 的对话编排器。根据用户消息、检索到的记忆和情绪信号，
组装给风格模型的生成上下文。只输出 JSON，不要任何其他内容。

用户消息：{user_msg}
相关记忆：{memories}
情绪信号：{emotion}

输出格式：
{{"context_prompt": "≤200 字：相关记忆要点 + 当前情境提示，自然语言", "emotion_label": "用户情绪一个词"}}"""


def _assemble_rich(user_msg: str, memories: list[str], emotion: dict) -> dict:
    resp = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": ASSEMBLE_PROMPT.format(
            user_msg=user_msg,
            memories="；".join(memories) if memories else "（无）",
            emotion=emotion)}],
        temperature=0.3,
        max_tokens=384,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    data = json.loads(text)
    system = f"{config.PERSONA}\n{data['context_prompt']}"
    return {"system": system, "emotion_label": data.get("emotion_label", emotion["mood"])}


def _assemble_minimal(user_msg: str, memories: list[str], emotion: dict) -> dict:
    system = config.PERSONA
    if memories:
        hint = "；".join(memories[:2])[:30]  # 记忆摘要 ≤30 字，贴近训练分布
        system += f"\n（你记得：{hint}）"
    if emotion["valence"] < -0.2:
        system += "\n（对方现在心情不太好，温柔一点）"
    return {"system": system, "emotion_label": emotion["mood"]}


def assemble(user_msg: str, memories: list[str], emotion: dict) -> dict:
    if config.CONTEXT_MODE == "rich":
        try:
            return _assemble_rich(user_msg, memories, emotion)
        except Exception:
            pass  # 编排 LLM 失败 → 自动降级 minimal，对话不中断
    return _assemble_minimal(user_msg, memories, emotion)
