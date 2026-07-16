# -*- coding: utf-8 -*-
"""server.py /chat — 端到端流程（全 mock，无外部依赖）"""
from fastapi.testclient import TestClient

import server


def _patch_pipeline(monkeypatch):
    monkeypatch.setattr(server, "detect_emotion",
                        lambda text: {"mood": "neutral", "valence": 0.0, "arousal": 0.2})
    monkeypatch.setattr(server, "search_memories",
                        lambda user_id, q: ["喵喵怕黑"])
    monkeypatch.setattr(server, "assemble",
                        lambda msg, mems, emo: {"system": "人格", "emotion_label": "neutral"})
    monkeypatch.setattr(server, "generate",
                        lambda system, msg, history=None: "抱抱嘛")
    monkeypatch.setattr(server, "extract_and_save",
                        lambda user_id, msg, reply: [])


def test_chat_response_shape(monkeypatch):
    _patch_pipeline(monkeypatch)
    client = TestClient(server.app)
    r = client.post("/chat", json={"message": "想你了"})
    assert r.status_code == 200
    data = r.json()
    assert data["reply"] == "抱抱嘛"
    assert data["emotion"] == "neutral"
    assert data["memories_used"] == 1
    assert data["context_mode"] in ("rich", "minimal")


def test_chat_history_trimmed_to_8(monkeypatch):
    _patch_pipeline(monkeypatch)
    seen = {}

    def spy_generate(system, msg, history=None):
        seen["history"] = history
        return "嗯嗯"

    monkeypatch.setattr(server, "generate", spy_generate)
    client = TestClient(server.app)
    long_history = [{"role": "user", "content": f"消息{i}"} for i in range(20)]
    client.post("/chat", json={"message": "在吗", "history": long_history})
    assert len(seen["history"]) == 8


def test_health(monkeypatch):
    client = TestClient(server.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
