# -*- coding: utf-8 -*-
"""generation/llama_client.py — 请求拼装、重试、鉴权头（mock httpx）"""
import httpx
import pytest

import config
from generation import llama_client


class _FakeResponse:
    def __init__(self, reply="哼，才没有想你"):
        self._reply = reply

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._reply}}]}


def test_generate_ok(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    reply = llama_client.generate("人格", "想你了", history=[
        {"role": "user", "content": "早"},
        {"role": "assistant", "content": "早呀"}])
    assert reply == "哼，才没有想你"
    msgs = captured["json"]["messages"]
    assert msgs[0] == {"role": "system", "content": "人格"}
    assert msgs[-1] == {"role": "user", "content": "想你了"}
    assert len(msgs) == 4  # system + 2 history + user


def test_retry_once_then_success(monkeypatch):
    calls = {"n": 0}

    def flaky_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("timeout")
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", flaky_post)
    assert llama_client.generate("人格", "在吗") == "哼，才没有想你"
    assert calls["n"] == 2


def test_both_attempts_fail_raises(monkeypatch):
    def dead_post(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", dead_post)
    with pytest.raises(RuntimeError, match="llama-server 不可用"):
        llama_client.generate("人格", "在吗")


def test_api_key_header(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(config, "LLAMA_API_KEY", "secret123")
    llama_client.generate("人格", "在吗")
    assert captured["headers"]["Authorization"] == "Bearer secret123"
