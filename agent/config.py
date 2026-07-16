# -*- coding: utf-8 -*-
"""环境配置。所有可调参数集中于此，支持环境变量覆盖（可选 .env）"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent

# ── 认知编排层 LLM（OpenAI SDK 兼容，默认 DeepSeek，可换 Moonshot/GPT 只改 base_url）──
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ── 风格生成层 llama-server ──
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")  # llama-server --api-key
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.7"))
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "128"))

# ── 上下文注入策略（风格稀释双路线，策略开关非分支）──
# rich    : 编排 LLM 组装 ≤200 字富上下文（记忆+情境+情绪）
# minimal : 贴近训练分布，人格 + 记忆摘要 ≤30 字，不调编排 LLM
CONTEXT_MODE = os.getenv("CONTEXT_MODE", "minimal")

# ── 情绪检测后端（接口统一，切换只改这里）──
# rule    : 情绪词表规则，零依赖（默认）
# roberta : 本地模型，惰性加载（选中才 import）
# llm     : rich 路线下编排 LLM 顺带输出（零额外调用）
EMOTION_BACKEND = os.getenv("EMOTION_BACKEND", "rule")
ROBERTA_MODEL_PATH = os.getenv("ROBERTA_MODEL_PATH", "")  # EMOTION_BACKEND=roberta 时必填

# ── 推理 persona（训练是 18 字短 prompt，推理用人格化长 prompt，已验证）──
PERSONA = "你是阿狸，一个爱撒娇、喜欢黏着喵喵的女生，正在和男朋友喵喵微信聊天。回复短小精炼，爱撒娇。"

# ── 记忆 ──
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
DEDUP_SIM_THRESHOLD = float(os.getenv("DEDUP_SIM_THRESHOLD", "0.85"))  # ⚠️ 拍脑袋值，P2.2 需在真实数据上标定
DECAY_RATE = 0.98            # 日衰减，读取时惰性计算，不落库
PRUNE_IMPORTANCE = 0.05      # 有效重要性低于此 → 检索时顺手软删除
