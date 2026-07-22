# -*- coding: utf-8 -*-
"""FastAPI 编排入口 — POST /chat 五步流程：情绪 → 检索 → 组装 → 生成 → 异步记忆写入

启动：cd agent && uvicorn server:app --host 0.0.0.0 --port 8000
浏览器打开 http://localhost:8000 即前端聊天页
"""
from fastapi import BackgroundTasks, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from context.assembler import assemble
from emotion.detector import detect_emotion
from generation.llama_client import generate
from memory.extract import extract_and_save
from memory.store import list_memories, search_memories

app = FastAPI(title="阿狸 Chatbot")


class ChatRequest(BaseModel):
    user_id: str = "喵喵"
    message: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


@app.post("/chat")
def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    emotion = detect_emotion(req.message)                       # 1. 情绪（本地 <10ms）
    memories = search_memories(req.user_id, req.message)        # 2. 检索（<100ms）
    ctx = assemble(req.message, memories, emotion)              # 3. 组装（minimal 0ms / rich ~1s）
    reply = generate(ctx["system"], req.message,                # 4. 风格生成
                     history=req.history[-8:])
    background_tasks.add_task(extract_and_save,                 # 5. 记忆写入（异步不阻塞）
                              req.user_id, req.message, reply)
    return {
        "reply": reply,
        "emotion": ctx["emotion_label"],
        "memories_used": len(memories),
        "context_mode": config.CONTEXT_MODE,
    }


@app.get("/user/{user_id}/memories")
def get_memories(user_id: str):
    """调试：查看某用户全部记忆及有效重要性"""
    return list_memories(user_id)


@app.get("/health")
def health():
    return {"status": "ok", "context_mode": config.CONTEXT_MODE,
            "emotion_backend": config.EMOTION_BACKEND}


# 前端挂根路径，必须放在所有路由之后
app.mount("/", StaticFiles(directory=str(config.BASE_DIR / "frontend"), html=True),
          name="frontend")
