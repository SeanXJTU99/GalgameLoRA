# -*- coding: utf-8 -*-
"""情绪检测 — 三后端统一接口，EMOTION_BACKEND 配置切换（rule/roberta/llm）

统一返回：{"mood": str, "valence": float[-1,1], "arousal": float[0,1]}
llm 后端在 assembler rich 路线里顺带完成，此模块直接透传 rule 结果作占位，
最终以编排 LLM 输出的 emotion_label 覆盖（见 context/assembler.py）。
"""
import config

# ── rule 后端：情绪词表 + 规则，零依赖（默认）──

_NEG = ["累", "难过", "烦", "哭", "生气", "唉", "呜", "委屈", "焦虑",
        "压力", "失眠", "痛", "难受", "孤独", "害怕", "emo", "崩溃"]
_POS = ["开心", "哈哈", "嘿嘿", "爱你", "想你", "喜欢", "嘻嘻", "耶",
        "棒", "好耶", "亲亲", "抱抱", "么么"]
_HIGH_AROUSAL = ["！", "!", "？？", "??", "气死", "崩溃", "太", "超级", "好耶"]


def _detect_rule(text: str) -> dict:
    neg = sum(w in text for w in _NEG)
    pos = sum(w in text for w in _POS)
    valence = max(-1.0, min(1.0, 0.3 * pos - 0.4 * neg))
    arousal = min(1.0, 0.2 + 0.2 * sum(w in text for w in _HIGH_AROUSAL))
    if valence < -0.2:
        mood = "negative"
    elif valence > 0.2:
        mood = "positive"
    else:
        mood = "neutral"
    return {"mood": mood, "valence": valence, "arousal": arousal}


# ── roberta 后端：本地模型，惰性加载（选中才 import，不装模型不报错）──

_roberta = None


def _detect_roberta(text: str) -> dict:
    global _roberta
    if _roberta is None:
        from transformers import pipeline  # 惰性 import
        if not config.ROBERTA_MODEL_PATH:
            raise RuntimeError("EMOTION_BACKEND=roberta 需要设置 ROBERTA_MODEL_PATH")
        _roberta = pipeline("text-classification",
                            model=config.ROBERTA_MODEL_PATH, top_k=1)
    result = _roberta(text)[0][0]
    label, score = result["label"].lower(), result["score"]
    valence = score if "pos" in label or label in ("happy", "like") else \
        -score if "neg" in label or label in ("sad", "angry", "fear") else 0.0
    return {"mood": label, "valence": valence, "arousal": score}


# ── 统一入口 ──

def detect_emotion(text: str) -> dict:
    if config.EMOTION_BACKEND == "roberta":
        try:
            return _detect_roberta(text)
        except Exception:
            return _detect_rule(text)  # 模型不可用 → 规则降级
    # rule 与 llm 后端都先走规则（llm 的最终标签由 assembler 覆盖）
    return _detect_rule(text)
