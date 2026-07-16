# -*- coding: utf-8 -*-
"""memory/store.py — 写入去重、检索排序、惰性衰减、用户隔离"""
from datetime import datetime, timedelta, timezone

import config
from memory.models import Memory


def test_add_and_search(mem_store):
    mem_store.add_memory("喵喵", Memory("喵喵怕黑，晚上不敢一个人走夜路", 0.7))
    results = mem_store.search_memories("喵喵", "晚上怕黑")
    assert "喵喵怕黑，晚上不敢一个人走夜路" in results


def test_dedup_same_content(mem_store):
    """相同内容写两次 → 合并不新增，保留更高 importance"""
    mem_store.add_memory("喵喵", Memory("喜欢吃山姆的蛋糕", 0.5))
    mem_store.add_memory("喵喵", Memory("喜欢吃山姆的蛋糕", 0.8))
    memories = mem_store.list_memories("喵喵")
    assert len(memories) == 1
    assert memories[0]["importance"] == 0.8


def test_user_isolation(mem_store):
    mem_store.add_memory("喵喵", Memory("喵喵的秘密", 0.9))
    assert mem_store.search_memories("别人", "秘密") == []


def test_effective_importance_decay(mem_store):
    """惰性衰减数学：30 天前的 0.5 → 0.5 × 0.98^30 ≈ 0.273"""
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    meta = {"importance": 0.5, "created_at": old, "last_recalled_at": old}
    imp = mem_store.effective_importance(meta)
    assert abs(imp - 0.5 * config.DECAY_RATE ** 30) < 0.01


def test_fresh_memory_no_decay(mem_store):
    mem_store.add_memory("喵喵", Memory("今天的事", 0.5))
    m = mem_store.list_memories("喵喵")[0]
    assert m["effective_importance"] > 0.49


def test_search_empty_store(mem_store):
    assert mem_store.search_memories("喵喵", "任何内容") == []
