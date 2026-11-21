from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import patch

import litellm
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.config import KeyEntry, Settings
from app.main import create_app


def _make_mock_response(content: str, prompt_tokens: int, completion_tokens: int, model: str = "gemini/gemini-2.0-flash") -> litellm.ModelResponse:
    return litellm.ModelResponse(
        id="chatcmpl-test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        model=model,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def _create_test_client(tmp_path, enable_sim: bool = True, keys: list[KeyEntry] | None = None) -> TestClient:
    db_file = tmp_path / "test.db"
    key_pool = keys if keys is not None else [
        KeyEntry(key_id="key_1", api_key="fake-key-1", model="gemini/gemini-2.0-flash"),
        KeyEntry(key_id="key_2", api_key="fake-key-2", model="gemini/gemini-2.0-flash"),
    ]
    settings = Settings(
        key_pool=key_pool,
        default_model="gemini/gemini-2.0-flash",
        default_system_prompt="You are a test assistant.",
        database_url=f"sqlite:///{db_file}",
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=enable_sim,
        cache_enabled=True,
    )
    app = create_app(settings=settings)
    app.state.gateway.clear_cache()
    return TestClient(app)


def test_health_endpoint(tmp_path):
    client = _create_test_client(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "keys_configured": 2,
        "cache_enabled": True,
    }


def test_chat_live_then_cache_hit_across_sessions(tmp_path):
    """Verify real LiteLLM built-in caching end-to-end across sessions."""
    keys = [
        KeyEntry(key_id="key_1", api_key="fake-key-1", model="openai/gpt-4o-mini"),
        KeyEntry(key_id="key_2", api_key="fake-key-2", model="openai/gpt-4o-mini"),
    ]
    client = _create_test_client(tmp_path, enable_sim=True, keys=keys)
    provider_calls: list[str] = []

    async def fake_provider(*args: Any, **kwargs: Any):
        provider_calls.append(kwargs.get("api_key", ""))
        await asyncio.sleep(0.06)  # 60ms simulated provider latency
        msgs = kwargs.get("data", {}).get("messages") or kwargs.get("messages") or []
        last_user = msgs[-1]["content"] if msgs else "unknown"
        return _make_mock_response(
            f"Answer for: {last_user}",
            prompt_tokens=42,
            completion_tokens=9,
            model="openai/gpt-4o-mini",
        )

    with patch.object(litellm.openai_chat_completions, "acompletion", side_effect=fake_provider):
        # 1st call -> LIVE_INFERENCE
        r1 = client.post(
            "/chat",
            json={"session_id": "s-cache-1", "message": "What is the capital of France?"},
        )
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["reply"] == "Answer for: What is the capital of France?"
        assert b1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert b1["metadata"]["key_id"] == "key_1"
        assert b1["metadata"]["fallback_attempts"] == 0
        assert b1["metadata"]["input_tokens"] == 42
        assert b1["metadata"]["output_tokens"] == 9
        assert b1["metadata"]["model"] == "openai/gpt-4o-mini"

        # 2nd call in a NEW session with the exact same first message -> LLM_CACHE
        r2 = client.post(
            "/chat",
            json={"session_id": "s-cache-2", "message": "What is the capital of France?"},
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["reply"] == b1["reply"]
        assert b2["metadata"]["resolution_source"] == "LLM_CACHE"
        assert b2["metadata"]["key_id"] is None
        assert b2["metadata"]["fallback_attempts"] == 0
        assert b2["metadata"]["input_tokens"] == 42
        assert b2["metadata"]["output_tokens"] == 9
        assert b2["metadata"]["latency_ms"] < b1["metadata"]["latency_ms"]

        # 3rd call with a DIFFERENT message -> must NOT come from cache
        r3 = client.post(
            "/chat",
            json={"session_id": "s-cache-3", "message": "What is the capital of Germany?"},
        )
        assert r3.status_code == 200
        b3 = r3.json()
        assert b3["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert b3["metadata"]["key_id"] == "key_1"

        # 4th call: same cached prompt WITH X-Simulate-Failure: key_1=429 must bypass cache and fail over to key_2
        r4 = client.post(
            "/chat",
            json={"session_id": "s-cache-4", "message": "What is the capital of France?"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert r4.status_code == 200
        b4 = r4.json()
        assert b4["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert b4["metadata"]["key_id"] == "key_2"
        assert b4["metadata"]["fallback_attempts"] == 1

    # Only 3 live provider calls for 4 requests (r2 was served from LiteLLM cache)
    assert provider_calls == ["fake-key-1", "fake-key-1", "fake-key-2"]

    # Check stored history for cache-hit session preserves key_id: null
    hist2 = client.get("/sessions/s-cache-2/messages")
    assert hist2.status_code == 200
    h2 = hist2.json()
    assert h2["messages"][1]["metadata"]["resolution_source"] == "LLM_CACHE"
    assert h2["messages"][1]["metadata"]["key_id"] is None


def test_validation_errors_return_422_and_do_not_persist(tmp_path):
    client = _create_test_client(tmp_path)

    # Empty message after trimming
    r1 = client.post("/chat", json={"session_id": "s-val-1", "message": "   "})
    assert r1.status_code == 422
    assert "detail" in r1.json()

    # Message over 4000 chars
    r2 = client.post("/chat", json={"session_id": "s-val-1", "message": "a" * 4001})
    assert r2.status_code == 422
    assert "detail" in r2.json()

    # Missing session_id
    r3 = client.post("/chat", json={"message": "hello"})
    assert r3.status_code == 422
    assert "detail" in r3.json()

    # Malformed JSON
    r4 = client.post("/chat", content="{bad json", headers={"Content-Type": "application/json"})
    assert r4.status_code == 422
    assert "detail" in r4.json()

    # Failed requests must not create session history
    r_hist = client.get("/sessions/s-val-1/messages")
    assert r_hist.status_code == 404
    assert r_hist.json() == {"detail": "session not found"}


def test_unknown_session_returns_404(tmp_path):
    client = _create_test_client(tmp_path)
    resp = client.get("/sessions/does-not-exist/messages")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "session not found"}


def test_simulate_failure_key1_failover_to_key2(tmp_path):
    client = _create_test_client(tmp_path, enable_sim=True)
    seen_keys: list[str] = []

    async def fake_acompletion(*args: Any, **kwargs: Any):
        seen_keys.append(kwargs.get("api_key", ""))
        return _make_mock_response("Recovered via key 2", 30, 8)

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        resp = client.post(
            "/chat",
            json={"session_id": "s-fail-1", "message": "Test failover 429"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s-fail-1"
    assert body["reply"] == "Recovered via key 2"
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["input_tokens"] == 30
    assert meta["output_tokens"] == 8
    assert seen_keys == ["fake-key-2"]

    # Next request WITHOUT X-Simulate-Failure must still use key_1 (no leaked cooldown)
    seen_keys.clear()
    with patch("litellm.acompletion", side_effect=fake_acompletion):
        resp2 = client.post(
            "/chat",
            json={"session_id": "s-after-sim", "message": "Normal request after sim"},
        )
    assert resp2.status_code == 200
    assert resp2.json()["metadata"]["key_id"] == "key_1"
    assert resp2.json()["metadata"]["fallback_attempts"] == 0
    assert seen_keys == ["fake-key-1"]


def test_simulate_failure_500_failover(tmp_path):
    client = _create_test_client(tmp_path, enable_sim=True)

    async def fake_acompletion(*args: Any, **kwargs: Any):
        return _make_mock_response("Recovered from 500", 25, 6)

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        resp = client.post(
            "/chat",
            json={"session_id": "s-fail-500", "message": "Test failover 500"},
            headers={"X-Simulate-Failure": "key_1=500"},
        )

    assert resp.status_code == 200
    meta = resp.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1


def test_simulate_failure_all_keys_returns_503_and_no_history(tmp_path):
    client = _create_test_client(tmp_path, enable_sim=True)

    resp = client.post(
        "/chat",
        json={"session_id": "s-all-fail", "message": "Should fail everywhere"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert resp.status_code == 503
    assert resp.json() == {"detail": "all providers unavailable"}

    hist = client.get("/sessions/s-all-fail/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}


def test_simulate_failure_ignored_when_disabled(tmp_path):
    client = _create_test_client(tmp_path, enable_sim=False)

    async def fake_acompletion(*args: Any, **kwargs: Any):
        return _make_mock_response("Ignored sim header", 20, 5)

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        resp = client.post(
            "/chat",
            json={"session_id": "s-sim-off", "message": "Sim disabled"},
            headers={"X-Simulate-Failure": "*=429"},
        )

    assert resp.status_code == 200
    assert resp.json()["metadata"]["key_id"] == "key_1"
    assert resp.json()["metadata"]["fallback_attempts"] == 0


def test_real_provider_429_triggers_automatic_failover(tmp_path):
    client = _create_test_client(tmp_path, enable_sim=True)

    async def fake_acompletion(*args: Any, **kwargs: Any):
        if kwargs.get("api_key") == "fake-key-1":
            raise litellm.RateLimitError(
                message="Quota exceeded",
                llm_provider="gemini",
                model="gemini/gemini-2.0-flash",
            )
        return _make_mock_response("Handled by key 2 after real 429", 33, 11)

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        resp = client.post(
            "/chat",
            json={"session_id": "s-real-429", "message": "Trigger real 429"},
        )

    assert resp.status_code == 200
    meta = resp.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"


def test_multi_turn_history_and_system_prompt_isolation(tmp_path):
    client = _create_test_client(tmp_path, enable_sim=True)
    captured_messages: list[list[dict[str, str]]] = []

    async def fake_acompletion(*args: Any, **kwargs: Any):
        msgs = kwargs.get("messages", [])
        captured_messages.append(list(msgs))
        user_last = msgs[-1]["content"]
        return _make_mock_response(f"Echo: {user_last}", len(msgs) * 10, 5)

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        r1 = client.post(
            "/chat",
            json={
                "session_id": "s-multi",
                "message": "My name is Alex.",
                "system_prompt": "Be concise.",
            },
        )
        assert r1.status_code == 200

        r2 = client.post(
            "/chat",
            json={"session_id": "s-multi", "message": "What is my name?"},
        )
        assert r2.status_code == 200

        # Separate session must not see s-multi history
        r3 = client.post(
            "/chat",
            json={"session_id": "s-other", "message": "What is my name?"},
        )
        assert r3.status_code == 200

    # Turn 1 messages: [system, user1]
    assert captured_messages[0] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "My name is Alex."},
    ]
    # Turn 2 messages: [system (persisted), user1, assistant1, user2]
    assert captured_messages[1] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "My name is Alex."},
        {"role": "assistant", "content": "Echo: My name is Alex."},
        {"role": "user", "content": "What is my name?"},
    ]
    # Other session has only default system prompt + its own message
    assert captured_messages[2] == [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "What is my name?"},
    ]

    # Verify GET /sessions/{session_id}/messages omits system prompt and preserves order
    hist = client.get("/sessions/s-multi/messages")
    assert hist.status_code == 200
    data = hist.json()
    assert data["session_id"] == "s-multi"
    assert len(data["messages"]) == 4
    assert data["messages"][0] == {"role": "user", "content": "My name is Alex."}
    assert data["messages"][1]["role"] == "assistant"
    assert data["messages"][1]["content"] == "Echo: My name is Alex."
    assert data["messages"][1]["metadata"]["input_tokens"] == 20
    assert data["messages"][2] == {"role": "user", "content": "What is my name?"}
    assert data["messages"][3]["role"] == "assistant"
