# -*- coding: utf-8 -*-
"""llama-server HTTP 客户端 — 走 /v1/chat/completions（OpenAI 兼容端点，
自动套用 GGUF 内嵌 chat template，无需手拼 ChatML）"""
import httpx

import config


def generate(system: str, user_msg: str, history: list[dict] = None,
             temperature: float = None, max_tokens: int = None) -> str:
    messages = [{"role": "system", "content": system}]
    messages += history or []
    messages.append({"role": "user", "content": user_msg})

    headers = {}
    if config.LLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLAMA_API_KEY}"
    payload = {
        "messages": messages,
        "temperature": temperature if temperature is not None else config.GEN_TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else config.GEN_MAX_TOKENS,
    }

    last_err = None
    for _ in range(2):  # 超时/瞬断重试一次
        try:
            r = httpx.post(f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
                           json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError) as e:
            last_err = e
    raise RuntimeError(f"llama-server 不可用: {last_err}")
