from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

# Prevent LiteLLM from blocking startup on remote GitHub cost-map fetches
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm import Router
from litellm.caching.caching import LiteLLMCacheType

from app.config import KeyEntry, Settings

logger = logging.getLogger(__name__)

VALID_SIMULATED_STATUSES = {429, 500, 502, 503}

# Silence noisy LiteLLM telemetry / debug banners
litellm.telemetry = False
litellm.suppress_debug_info = True


class AllProvidersUnavailableError(Exception):
    """Raised when every configured key in the pool fails."""


@dataclass
class _RequestTrace:
    simulated_failures: dict[str, int] = field(default_factory=dict)
    attempts: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    winning_key_id: str | None = None
    winning_model: str | None = None


_current_trace: contextvars.ContextVar[_RequestTrace | None] = contextvars.ContextVar(
    "lyftr_request_trace", default=None
)


def parse_simulate_failure_header(header_value: str | None) -> dict[str, int]:
    """
    Parse X-Simulate-Failure header into a {key_id_or_star: status_code} map.
    Examples:
      "key_1=429" -> {"key_1": 429}
      "key_1=429,key_2=500" -> {"key_1": 429, "key_2": 500}
      "*=429" -> {"*": 429}
    """
    if not header_value:
        return {}

    parsed: dict[str, int] = {}
    for part in header_value.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key_id, raw_status = part.split("=", 1)
        key_id = key_id.strip()
        raw_status = raw_status.strip()
        if not key_id or not raw_status.isdigit():
            continue
        status = int(raw_status)
        if status in VALID_SIMULATED_STATUSES:
            parsed[key_id] = status
    return parsed


def _build_simulated_exception(status_code: int, model: str, key_id: str) -> Exception:
    provider = model.split("/", 1)[0] if "/" in model else "custom"
    msg = f"Simulated HTTP {status_code} failure for {key_id}"
    if status_code == 429:
        return litellm.RateLimitError(message=msg, llm_provider=provider, model=model)
    if status_code == 502:
        return litellm.BadGatewayError(message=msg, llm_provider=provider, model=model)
    if status_code == 503:
        return litellm.ServiceUnavailableError(message=msg, llm_provider=provider, model=model)
    return litellm.InternalServerError(message=msg, llm_provider=provider, model=model)


def _is_retryable_provider_error(exc: Exception) -> bool:
    """Return True for 429, quota, 5xx, timeout, or connection errors."""
    retryable_types = (
        litellm.RateLimitError,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
        litellm.BadGatewayError,
        litellm.Timeout,
        litellm.APIConnectionError,
        litellm.APIError,
    )
    if isinstance(exc, retryable_types):
        status = getattr(exc, "status_code", None)
        if status is None or status == 429 or (500 <= int(status) < 600):
            return True
    status = getattr(exc, "status_code", None)
    if status is not None and (int(status) == 429 or 500 <= int(status) < 600):
        return True
    msg = str(exc).lower()
    if any(tok in msg for tok in ("429", "quota", "rate limit", "resource_exhausted", "500", "502", "503", "504")):
        return True
    return False


@dataclass
class GatewayResult:
    reply: str
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


class _FailoverRouter(Router):
    """
    LiteLLM Router subclass that:
    1. Injects per-request X-Simulate-Failure errors cleanly via ContextVar without mutating router state.
    2. Records per-request key attempts and failures for accurate key_id and fallback_attempts reporting.
    """

    async def _acompletion(self, *args: Any, **kwargs: Any) -> Any:
        trace = _current_trace.get()
        # In LiteLLM Router, model_name passed to _acompletion is the deployment's model_name (our key_id)
        requested_model_name = kwargs.get("model") or (args[0] if args else None)
        key_id = str(requested_model_name) if requested_model_name else None

        deployment = self._find_deployment_by_name(key_id) if key_id else None
        actual_model = (
            deployment["litellm_params"]["model"]
            if deployment is not None
            else str(requested_model_name or "")
        )

        if trace is not None and key_id is not None:
            trace.attempts.append(key_id)
            sim_status = trace.simulated_failures.get(key_id) or trace.simulated_failures.get("*")
            if sim_status is not None:
                trace.failures.append(key_id)
                raise _build_simulated_exception(sim_status, actual_model, key_id)

        try:
            response = await super()._acompletion(*args, **kwargs)
            if trace is not None and key_id is not None:
                trace.winning_key_id = key_id
                trace.winning_model = actual_model
            return response
        except Exception:
            if trace is not None and key_id is not None:
                trace.failures.append(key_id)
            raise

    def _find_deployment_by_name(self, model_name: str | None) -> dict[str, Any] | None:
        if not model_name:
            return None
        for item in self.model_list:
            if item.get("model_name") == model_name:
                return item
        return None


class LLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key_pool: list[KeyEntry] = list(settings.key_pool)
        self.key_map: dict[str, KeyEntry] = {entry.key_id: entry for entry in self.key_pool}

        # Enable LiteLLM's built-in in-memory cache
        litellm.enable_cache(type=LiteLLMCacheType.LOCAL)
        self.cache_enabled = True

        self.router: _FailoverRouter | None = self._build_router()

    def _build_router(self) -> _FailoverRouter | None:
        if not self.key_pool:
            return None

        model_list = [
            {
                "model_name": entry.key_id,
                "litellm_params": {
                    "model": entry.model,
                    "api_key": entry.api_key,
                },
                "model_info": {
                    "id": entry.key_id,
                },
            }
            for entry in self.key_pool
        ]

        fallbacks: list[dict[str, list[str]]] = []
        if len(self.key_pool) > 1:
            primary_id = self.key_pool[0].key_id
            fallback_ids = [entry.key_id for entry in self.key_pool[1:]]
            fallbacks = [{primary_id: fallback_ids}]

        return _FailoverRouter(
            model_list=model_list,
            fallbacks=fallbacks,
            cache_responses=True,
            cache_kwargs={"type": "local"},
            num_retries=0,
            allowed_fails=1_000_000,
            cooldown_time=0,
            set_verbose=False,
        )

    @property
    def keys_configured(self) -> int:
        return len(self.key_pool)

    def clear_cache(self) -> None:
        if litellm.cache is not None and hasattr(litellm.cache, "flush_cache"):
            litellm.cache.flush_cache()

    async def complete(
        self,
        messages: list[dict[str, str]],
        simulate_failure_header: str | None = None,
    ) -> GatewayResult:
        if not self.key_pool or self.router is None:
            raise AllProvidersUnavailableError("No API keys configured")

        simulated_failures: dict[str, int] = {}
        if self.settings.enable_failure_simulation and simulate_failure_header:
            simulated_failures = parse_simulate_failure_header(simulate_failure_header)

        # When failure simulation is active for this request, bypass cache read
        # so the failover path is guaranteed to execute on demand.
        use_cache = self.cache_enabled and not bool(simulated_failures)

        trace = _RequestTrace(simulated_failures=simulated_failures)
        token = _current_trace.set(trace)
        start = time.perf_counter()

        try:
            primary_entry = self.key_pool[0]
            call_kwargs: dict[str, Any] = {
                "model": primary_entry.key_id,
                "messages": messages,
                "caching": use_cache,
            }
            if not use_cache:
                call_kwargs["cache"] = {"no-cache": True, "no-store": True}

            response = await self.router.acompletion(**call_kwargs)
        except Exception as exc:
            logger.warning("All LLM keys failed: %s", type(exc).__name__)
            raise AllProvidersUnavailableError("all providers unavailable") from exc
        finally:
            _current_trace.reset(token)

        latency_ms = int(round((time.perf_counter() - start) * 1000))

        hidden = getattr(response, "_hidden_params", None) or {}
        is_cache_hit = bool(hidden.get("cache_hit", False))

        # Let LiteLLM's background async cache write flush before returning
        if use_cache and not is_cache_hit:
            await asyncio.sleep(0)

        reply_text = self._extract_reply_text(response)
        if not reply_text:
            raise AllProvidersUnavailableError("Empty response from provider")

        input_tokens, output_tokens = self._extract_token_usage(response)

        if is_cache_hit:
            resolution_source = "LLM_CACHE"
            key_id: str | None = None
            fallback_attempts = 0
            resolved_model = trace.winning_model or self.key_pool[0].model
        else:
            resolution_source = "LIVE_INFERENCE"
            key_id = trace.winning_key_id or hidden.get("model_id") or self.key_pool[0].key_id
            fallback_attempts = len(trace.failures)
            if key_id in self.key_map:
                resolved_model = self.key_map[key_id].model
            else:
                resolved_model = trace.winning_model or self.key_pool[0].model

        return GatewayResult(
            reply=reply_text,
            resolution_source=resolution_source,
            model=resolved_model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            key_id=key_id,
            fallback_attempts=fallback_attempts,
        )

    @staticmethod
    def _extract_reply_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        return ""

    @staticmethod
    def _extract_token_usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            return prompt_tokens, completion_tokens
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return prompt_tokens, completion_tokens
