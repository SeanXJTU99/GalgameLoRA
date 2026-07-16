# -*- coding: utf-8 -*-
"""emotion/detector.py — rule 后端 + roberta 降级"""
import config
from emotion.detector import detect_emotion


def test_negative():
    r = detect_emotion("今天好累啊，压力好大，想哭")
    assert r["mood"] == "negative"
    assert r["valence"] < -0.2


def test_positive():
    r = detect_emotion("嘿嘿开心，爱你哦")
    assert r["mood"] == "positive"
    assert r["valence"] > 0.2


def test_neutral():
    r = detect_emotion("晚饭吃什么")
    assert r["mood"] == "neutral"


def test_valence_clamped():
    r = detect_emotion("累烦哭生气委屈焦虑压力失眠痛难受孤独害怕崩溃")
    assert r["valence"] >= -1.0


def test_roberta_backend_falls_back_to_rule(monkeypatch):
    """选 roberta 但模型不可用 → 规则降级不报错"""
    monkeypatch.setattr(config, "EMOTION_BACKEND", "roberta")
    monkeypatch.setattr(config, "ROBERTA_MODEL_PATH", "")
    r = detect_emotion("今天好累")
    assert r["mood"] == "negative"
