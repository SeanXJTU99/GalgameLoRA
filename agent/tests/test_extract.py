# -*- coding: utf-8 -*-
"""memory/extract.py — JSON 解析健壮性（mock LLM）"""
import json

from memory import extract


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


def _with(monkeypatch, content):
    monkeypatch.setattr(extract, "_get_client", lambda: _FakeLLM(content))


def test_valid_json(monkeypatch):
    _with(monkeypatch, json.dumps(
        [{"content": "喵喵怕黑", "importance": 0.7}], ensure_ascii=False))
    ms = extract.extract_memories("我怕黑", "抱抱")
    assert len(ms) == 1
    assert ms[0].content == "喵喵怕黑"
    assert ms[0].importance == 0.7


def test_fenced_json(monkeypatch):
    _with(monkeypatch, "```json\n[{\"content\": \"喜欢蛋糕\"}]\n```")
    ms = extract.extract_memories("买了蛋糕", "好耶")
    assert len(ms) == 1
    assert ms[0].importance == 0.5  # 缺省值


def test_empty_array(monkeypatch):
    _with(monkeypatch, "[]")
    assert extract.extract_memories("在吗", "在") == []


def test_invalid_json_returns_empty(monkeypatch):
    _with(monkeypatch, "抱歉我无法提取")
    assert extract.extract_memories("在吗", "在") == []


def test_extract_and_save_llm_error_silent(monkeypatch):
    def boom():
        raise RuntimeError("API down")
    monkeypatch.setattr(extract, "_get_client", boom)
    assert extract.extract_and_save("喵喵", "在吗", "在") == []  # 静默不抛
