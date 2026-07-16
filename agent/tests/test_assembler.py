# -*- coding: utf-8 -*-
"""context/assembler.py — 双路线：minimal 纯函数 + rich mock LLM + 降级"""
import json

import config
from context import assembler


def test_minimal_plain():
    r = assembler._assemble_minimal("在吗", [], {"mood": "neutral", "valence": 0.0})
    assert r["system"] == config.PERSONA
    assert r["emotion_label"] == "neutral"


def test_minimal_with_memories_truncated():
    memories = ["喵喵怕黑，晚上不敢一个人走夜路", "喜欢吃山姆的蛋糕", "第三条不该出现"]
    r = assembler._assemble_minimal("晚上回家", memories, {"mood": "neutral", "valence": 0.0})
    assert "你记得" in r["system"]
    assert "第三条" not in r["system"]  # 只取 top-2
    hint = r["system"].split("你记得：")[1].split("）")[0]
    assert len(hint) <= 30  # 摘要 ≤30 字


def test_minimal_negative_adds_gentle_hint():
    r = assembler._assemble_minimal("好累", [], {"mood": "negative", "valence": -0.5})
    assert "温柔一点" in r["system"]


class _FakeLLM:
    def __init__(self, content):
        self._content = content
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        class Msg:
            pass
        msg = Msg()
        msg.content = self._content
        choice = Msg()
        choice.message = msg
        resp = Msg()
        resp.choices = [choice]
        return resp


def test_rich_parses_fenced_json(monkeypatch):
    fenced = "```json\n" + json.dumps(
        {"context_prompt": "记得对方怕黑", "emotion_label": "scared"},
        ensure_ascii=False) + "\n```"
    monkeypatch.setattr(assembler, "_get_client", lambda: _FakeLLM(fenced))
    monkeypatch.setattr(config, "CONTEXT_MODE", "rich")
    r = assembler.assemble("晚上一个人", ["喵喵怕黑"], {"mood": "neutral", "valence": 0.0})
    assert "记得对方怕黑" in r["system"]
    assert r["system"].startswith(config.PERSONA)
    assert r["emotion_label"] == "scared"


def test_rich_failure_falls_back_to_minimal(monkeypatch):
    def boom():
        raise RuntimeError("API down")
    monkeypatch.setattr(assembler, "_get_client", boom)
    monkeypatch.setattr(config, "CONTEXT_MODE", "rich")
    r = assembler.assemble("在吗", [], {"mood": "neutral", "valence": 0.0})
    assert r["system"] == config.PERSONA  # 降级为 minimal，对话不中断
