# -*- coding: utf-8 -*-
"""共享 fixtures — 伪 embedding 避免下载 BGE，全部测试可离线跑"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))  # agent/ 根


class FakeEF:
    """确定性伪 embedding：字符 bigram 哈希到 64 维。
    相同文本 → 相同向量；重叠多的文本 → 相似向量。"""

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def name():
        return "fake"

    def is_legacy(self):
        return False

    def __call__(self, input):
        return [self._embed(t) for t in input]

    def embed_query(self, input):  # chromadb 0.6+ 查询侧接口
        return self(input)

    @staticmethod
    def _embed(text):
        v = [0.0] * 64
        for i in range(len(text)):
            v[hash(text[i:i + 2]) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


@pytest.fixture()
def mem_store(tmp_path, monkeypatch):
    """隔离的记忆库：临时目录 + 伪 embedding，每个测试独立"""
    import config
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    from chromadb.utils import embedding_functions
    monkeypatch.setattr(embedding_functions,
                        "SentenceTransformerEmbeddingFunction", FakeEF)
    import memory.store as store
    store._collection = None
    yield store
    store._collection = None
