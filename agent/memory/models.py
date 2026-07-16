# -*- coding: utf-8 -*-
"""Memory 数据模型"""
from dataclasses import dataclass


@dataclass
class Memory:
    content: str            # "喵喵怕黑，晚上不敢一个人走夜路"
    importance: float = 0.5  # 0-1：强烈情绪 0.7-0.9，偏好陈述 0.6，普通事实 0.5
