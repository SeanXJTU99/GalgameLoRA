# -*- coding: utf-8 -*-
"""ChromaDB 记忆存储：写入去重、检索排序、惰性衰减（不落库，读取时现算）"""
import uuid
from datetime import datetime, timezone

import chromadb
from chromadb.utils import embedding_functions

import config
from .models import Memory

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL)
        _collection = client.get_or_create_collection(
            "memories", embedding_function=ef,
            metadata={"hnsw:space": "cosine"})
    return _collection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_since(iso: str) -> float:
    then = datetime.fromisoformat(iso)
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400)


def effective_importance(meta: dict) -> float:
    """惰性衰减：importance × 0.98^天数，无 cron 无定时器，天然幂等"""
    days = _days_since(meta.get("last_recalled_at") or meta["created_at"])
    return meta["importance"] * (config.DECAY_RATE ** days)


def add_memory(user_id: str, memory: Memory) -> str:
    """写入 + 去重：与最相似旧记忆 sim > 阈值 → 合并（保留更高 importance，刷新时间），不新增"""
    col = _get_collection()
    if col.count() > 0:
        res = col.query(query_texts=[memory.content], n_results=1,
                        where={"user_id": user_id})
        if res["ids"][0]:
            sim = 1 - res["distances"][0][0]
            if sim > config.DEDUP_SIM_THRESHOLD:
                old_id = res["ids"][0][0]
                old_meta = res["metadatas"][0][0]
                col.update(ids=[old_id], metadatas=[{
                    **old_meta,
                    "importance": max(old_meta["importance"], memory.importance),
                    "last_recalled_at": _now(),
                }])
                return old_id
    mem_id = str(uuid.uuid4())
    col.add(ids=[mem_id], documents=[memory.content], metadatas=[{
        "user_id": user_id,
        "importance": memory.importance,
        "created_at": _now(),
        "last_recalled_at": _now(),
        "recall_count": 0,
        "deleted": False,
    }])
    return mem_id


def search_memories(user_id: str, query: str, k: int = None) -> list[str]:
    """检索 → 相似度 × 有效重要性 排序 top-k；低于剪枝阈值的顺手软删除"""
    k = k or config.MEMORY_TOP_K
    col = _get_collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query],
                    n_results=min(k * 3, col.count()),
                    where={"user_id": user_id})
    scored = []
    for mem_id, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                       res["metadatas"][0], res["distances"][0]):
        if meta.get("deleted"):
            continue
        imp = effective_importance(meta)
        if imp < config.PRUNE_IMPORTANCE:
            col.update(ids=[mem_id], metadatas=[{**meta, "deleted": True}])
            continue
        scored.append(((1 - dist) * imp, mem_id, doc, meta))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:k]
    for _, mem_id, _, meta in top:  # 召回刷新，供衰减计算
        col.update(ids=[mem_id], metadatas=[{
            **meta,
            "recall_count": meta["recall_count"] + 1,
            "last_recalled_at": _now(),
        }])
    return [doc for _, _, doc, _ in top]


def list_memories(user_id: str) -> list[dict]:
    """调试用：返回该用户全部未删除记忆及有效重要性"""
    col = _get_collection()
    res = col.get(where={"user_id": user_id})
    out = []
    for doc, meta in zip(res["documents"], res["metadatas"]):
        if meta.get("deleted"):
            continue
        out.append({"content": doc,
                    "importance": meta["importance"],
                    "effective_importance": round(effective_importance(meta), 4),
                    "recall_count": meta["recall_count"],
                    "created_at": meta["created_at"]})
    out.sort(key=lambda m: m["effective_importance"], reverse=True)
    return out
