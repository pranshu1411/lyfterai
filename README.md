# LLM Chatbot with Key Failover and Response Caching

Full-stack LLM chatbot built with React (Vite), FastAPI, LiteLLM, and SQLite. Supports multi-turn session history, deterministic multi-key failover (`X-Simulate-Failure` test hook included), built-in LiteLLM response caching, and per-request token/latency reporting.

## 1. Live URLs

- **Frontend:** https://lyfterai-ochre.vercel.app
- **Backend:** https://lyfterai.onrender.com
- **Health Check:** https://lyfterai.onrender.com/health

## 2. Run locally

### With Docker (recommended)

1. Copy `.env.example` to `.env` and fill in at least two API keys in `LLM_API_KEYS`:
   ```bash
   cp .env.example .env
   ```
2. Start the full stack:
   ```bash
   make up
   ```
   - Backend runs at `http://localhost:8000`
   - Frontend runs at `http://localhost:3000`
   - SQLite persists on the named Docker volume mounted at `/data/app.db`
3. Stop the stack:
   ```bash
   make down
   ```

### Without Docker

**Backend:**
```bash
cd backend
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

export LLM_API_KEYS="key1,key2"
export LLM_MODEL="gemini/gemini-2.0-flash"
export DATABASE_URL="sqlite:///./app.db"
export ALLOWED_ORIGINS="http://localhost:3000"
export ENABLE_FAILURE_SIMULATION="true"

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

**Backend tests:**
```bash
cd backend
pytest -v
```

## 3. Design decisions

- **Key fallback (`litellm.Router`):** Keys from `LLM_API_KEYS` are registered in exact order (`key_1`, `key_2`, ...) as LiteLLM Router deployments with an explicit ordered fallback chain (`key_1 -> key_2 -> ...`). `num_retries=0`, `allowed_fails=1_000_000`, and `cooldown_time=0` prevent single-request simulated failures (`X-Simulate-Failure`) from poisoning subsequent normal requests, while still failing over immediately on real HTTP 429, quota, or 5xx responses. Per-request attempt/failure tracking uses `contextvars.ContextVar` so concurrent requests stay isolated.
- **Caching (LiteLLM built-in `Cache`):** Enabled LiteLLM's built-in in-memory `Cache(type="local")`. Cache keys are computed by LiteLLM from the model and the complete message list (`system + history + new message`), so identical first prompts across new sessions hit `LLM_CACHE` with `key_id: null`, `fallback_attempts: 0`, and preserved token counts, while differing messages or histories always miss. When `X-Simulate-Failure` is active (and `ENABLE_FAILURE_SIMULATION=true`), cache reads are bypassed for that request so failover behavior is deterministically exercised.
- **SQLite session storage:** Uses Python's standard library `sqlite3` (with `busy_timeout=5000`) and two tables (`sessions` and `messages`) created automatically on startup under `DATABASE_URL` (`/data/app.db` in Docker). A user/assistant turn is committed atomically in a single transaction only after a successful LLM response, ensuring failed `422` or `503` requests never pollute conversation history.

## 4. Verification

### Health check
```bash
curl http://localhost:8000/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}
```

### Live call & cache hit (across two sessions)
```bash
# 1st call -> LIVE_INFERENCE
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-cache-1","message":"What is the capital of France?"}'

# 2nd call in a new session with identical prompt -> LLM_CACHE (faster latency_ms, key_id: null)
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-cache-2","message":"What is the capital of France?"}'
```

### Failover via `X-Simulate-Failure`
```bash
# Fail key_1 with 429 -> 200 answered by key_2 with fallback_attempts: 1
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s-fail-1","message":"Name three primary colors."}'

# Fail every key -> 503 {"detail":"all providers unavailable"}, nothing saved to SQLite
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s-fail-all","message":"This should fail."}'
```

### Multi-turn history & per-message token usage
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-turn","message":"My favorite language is Python."}'

curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-turn","message":"What is my favorite language?"}'

curl -s http://localhost:8000/sessions/s-turn/messages
```
Each assistant reply in the UI displays `LIVE_INFERENCE` / `LLM_CACHE`, model, latency (`ms`), and per-request input/output tokens.

## 5. Setup Used

- **Editor:** Antigravity IDE
- **AI Tools:** Antigravity coding assistant

## 6. Bonus features (if any)

- **Session system prompt (`system-prompt-input`):** Optional input in the UI that sets and persists a custom system prompt per session in SQLite while keeping `GET /sessions/{session_id}/messages` clean of system messages.
- **Per-key model override (`LLM_MODELS`):** Optional comma-separated model list aligned with `LLM_API_KEYS`, allowing failover across keys from different models or providers when desired.
