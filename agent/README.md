# 阿狸 Chatbot

记忆增强型风格聊天机器人（P2）— 编排层（DeepSeek）+ 记忆层（ChromaDB + BGE）+ 风格层（v4 LoRA GGUF + llama-server）。

## 快速启动

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 启动风格层（本地 CPU 模式示例；GPU 改 --n-gpu-layers 99）
llama-server -m models/qwen7b_chatstyle_Q4_K_M.gguf --port 8080 --n-gpu-layers 0

# 3. 配置（或写 .env）
export LLM_API_KEY=sk-xxx           # DeepSeek key
export LLAMA_SERVER_URL=http://127.0.0.1:8080

# 4. 启动
uvicorn server:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://localhost:8000` 聊天（手机用局域网 IP）。

## 配置开关

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `CONTEXT_MODE` | `minimal` | 风格稀释双路线：`rich`（LLM 组装 ≤200 字）/ `minimal`（人格+记忆摘要 ≤30 字） |
| `EMOTION_BACKEND` | `rule` | 情绪检测：`rule` / `roberta`（需 `ROBERTA_MODEL_PATH`）/ `llm` |
| `LLM_BASE_URL` | DeepSeek | OpenAI 兼容编排 LLM，换 Moonshot/GPT 只改此项 |
| `DEDUP_SIM_THRESHOLD` | 0.85 | 记忆去重阈值 ⚠️ 需真实数据标定（P2.2） |

## API

```
POST /chat                      {user_id, message, history} → {reply, emotion, memories_used, context_mode}
GET  /user/{user_id}/memories   调试：查看记忆及有效重要性
GET  /health
```

## 评估与维护

```bash
python eval/style_baseline.py --v4-url http://localhost:8080 --out baseline.jsonl   # 风格基线
python scripts/backup_memory.py                                                      # 记忆备份
```
