"""
Self-hosted LLM client with an OpenRouter fallback.

Talks to an in-cluster / local OpenAI-compatible ``/v1`` endpoint (vLLM, TGI,
llama.cpp --server, Ollama, LM Studio) instead of a paid API, and degrades to
OpenRouter when the self-hosted service is unreachable or keeps erroring.

Ported from regulatory-explorer's ``processing/llm_service_client.py`` so the
two repos behave identically against the same GPU, with three changes for this
pipeline: the fallback client is Stage C's own OpenRouter client
(``ai_extraction._client``) rather than ``matchers/base``, JSON parsing reuses
``processing/json_utils.py``, and ``call_json`` returns the provider that
served each call so a benchmark can attribute per-provider cost.

Environment
-----------
    LLM_SERVICE_BASE_URL               e.g. http://localhost:8000/v1
    LLM_SERVICE_API_KEY                may be empty for an unauthenticated service
    LLM_SERVICE_MODEL                  e.g. gemma-4-26b
    LLM_SERVICE_CONTEXT_TOKENS         context window in tokens (e.g. 22000)
    LLM_SERVICE_DISCOVERY_CONCURRENCY  max parallel requests (default 8)
    LLM_SERVICE_TIMEOUT                per-request seconds (default 180)
    LLM_SERVICE_RETRIES                attempts per provider (default 3)
    LLM_SERVICE_FALLBACK               'false' makes self-hosted failures fatal
    LLM_FALLBACK_MODEL                 OpenRouter model for the fallback
    LLM_SERVICE_FALLBACK_AFTER         consecutive failures before sticking (default 2)

Usage
-----
    from processing.llm_service_client import health_check, call_json, provider_stats

    health_check()                       # aborts early if no provider answers
    obj, provider, meta = call_json(system, user, max_tokens=8192)
"""
import logging
import os
import threading
import time

import openai

from processing.json_utils import parse_llm_json, JsonParseError

logger = logging.getLogger(__name__)

LLM_SERVICE_BASE_URL = (os.getenv("LLM_SERVICE_BASE_URL", "") or "").rstrip("/")
LLM_SERVICE_API_KEY = os.getenv("LLM_SERVICE_API_KEY", "") or ""
LLM_SERVICE_MODEL = os.getenv("LLM_SERVICE_MODEL", "gemma-4-26b")
LLM_SERVICE_CONTEXT_TOKENS = int(os.getenv("LLM_SERVICE_CONTEXT_TOKENS", "22000"))
LLM_SERVICE_CONCURRENCY = int(os.getenv("LLM_SERVICE_DISCOVERY_CONCURRENCY", "8"))
LLM_SERVICE_TIMEOUT = int(os.getenv("LLM_SERVICE_TIMEOUT", "180"))
LLM_SERVICE_RETRIES = int(os.getenv("LLM_SERVICE_RETRIES", "3"))

# On by default so a dead endpoint degrades a long run instead of aborting it —
# but the fallback is a PAID API, unlike the self-hosted GPU, so it logs loudly.
LLM_FALLBACK_ENABLED = os.getenv("LLM_SERVICE_FALLBACK", "true").lower() == "true"
LLM_FALLBACK_MODEL = os.getenv(
    "LLM_FALLBACK_MODEL",
    os.getenv("EXTRACTOR_MODEL", "google/gemini-2.5-flash"),
)
# Consecutive self-hosted failures before the run stops preferring it. Without
# this every product pays the full retry/backoff cost against a dead endpoint.
LLM_FALLBACK_AFTER_FAILURES = int(os.getenv("LLM_SERVICE_FALLBACK_AFTER", "2"))

# How long to stay on the fallback before re-probing the self-hosted endpoint.
# Degradation used to be permanent for the run, which meant a 90-second
# llm-service restart pushed an entire multi-hour run onto the paid API and it
# never came back. The cooldown doubles on each failed probe up to
# LLM_SERVICE_RECOVERY_MAX, so a genuinely dead endpoint still stops costing a
# retry per product, while a transient blip costs only one cooldown.
LLM_SERVICE_RECOVERY_SECONDS = int(os.getenv("LLM_SERVICE_RECOVERY_AFTER", "60"))
LLM_SERVICE_RECOVERY_MAX = int(os.getenv("LLM_SERVICE_RECOVERY_MAX", "900"))

SELF_HOSTED = "self_hosted"
OPENROUTER = "openrouter"

_client = None
_client_lock = threading.Lock()
# Caps in-flight requests so a wide worker pool cannot overrun the self-hosted
# server's batch capacity.
_semaphore = threading.Semaphore(LLM_SERVICE_CONCURRENCY)

_provider_lock = threading.Lock()
_degraded = False            # True while the run prefers the fallback
_consecutive_failures = 0
# Monotonic deadline after which ONE caller may re-probe the self-hosted
# endpoint. Only one probe runs at a time (_probe_in_flight): with several
# workers, letting them all retry a dead endpoint simultaneously would pay the
# full retry/backoff cost N times over for no extra information.
_recovery_at = 0.0
_recovery_delay = 0.0
_probe_in_flight = False
_call_counts = {SELF_HOSTED: 0, OPENROUTER: 0}
_token_counts = {
    SELF_HOSTED: {"prompt": 0, "completion": 0},
    OPENROUTER: {"prompt": 0, "completion": 0, "cost": 0.0},
}


class LLMServiceError(RuntimeError):
    """Raised when no provider can serve the request."""


def get_llm_service_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not LLM_SERVICE_BASE_URL:
                    raise LLMServiceError(
                        "LLM_SERVICE_BASE_URL is not set — cannot reach the "
                        "self-hosted LLM (see health_check for the fallback path)."
                    )
                _client = openai.OpenAI(
                    base_url=LLM_SERVICE_BASE_URL,
                    # The SDK rejects an empty key; local servers ignore it.
                    api_key=LLM_SERVICE_API_KEY or "not-needed",
                    timeout=LLM_SERVICE_TIMEOUT,
                    max_retries=0,      # retries handled below, with backoff
                )
    return _client


def get_fallback_client():
    """
    Stage C's own OpenRouter client, imported lazily so a run that never needs
    the fallback does not require OPEN_ROUTER_API_KEY to be present.
    """
    from processing.ai_extraction import _client as openrouter_client
    return openrouter_client()


def fallback_available():
    if not LLM_FALLBACK_ENABLED:
        return False
    return bool(os.getenv("OPEN_ROUTER_API_KEY"))


def current_provider():
    """
    Which provider the next call should use.

    Self-hosted is always preferred. While degraded the fallback is used, EXCEPT
    that the first caller past the cooldown is elected to re-probe self-hosted —
    so a recovered endpoint is picked back up automatically instead of the run
    finishing on the paid API.
    """
    global _probe_in_flight
    if not LLM_SERVICE_BASE_URL:
        return OPENROUTER
    with _provider_lock:
        if not _degraded:
            return SELF_HOSTED
        if not _probe_in_flight and time.monotonic() >= _recovery_at:
            _probe_in_flight = True
            logger.info(
                f"[llm_service] re-probing self-hosted LLM after "
                f"{_recovery_delay:.0f}s on the fallback"
            )
            return SELF_HOSTED
        return OPENROUTER


def provider_stats():
    """Per-provider call/token counts, for an end-of-run summary."""
    with _provider_lock:
        return {
            "calls": dict(_call_counts),
            "tokens": {k: dict(v) for k, v in _token_counts.items()},
            "degraded": _degraded,
            "recovery_in_s": (max(0.0, _recovery_at - time.monotonic())
                              if _degraded else None),
        }


def reset_stats():
    global _degraded, _consecutive_failures, _recovery_at, _recovery_delay
    global _probe_in_flight
    with _provider_lock:
        _degraded = False
        _consecutive_failures = 0
        _recovery_at = 0.0
        _recovery_delay = 0.0
        _probe_in_flight = False
        for provider in _call_counts:
            _call_counts[provider] = 0
        _token_counts[SELF_HOSTED].update(prompt=0, completion=0)
        _token_counts[OPENROUTER].update(prompt=0, completion=0, cost=0.0)


def _mark_failure():
    """
    Record a self-hosted failure; return True if the run should now prefer the
    fallback.

    Not sticky any more. Degrading arms a cooldown, after which one caller
    re-probes self-hosted (see current_provider). A failed probe doubles the
    cooldown, so a dead endpoint converges on "checked rarely" rather than
    "never checked again".
    """
    global _degraded, _consecutive_failures, _recovery_at, _recovery_delay
    global _probe_in_flight
    with _provider_lock:
        _consecutive_failures += 1
        was_probing, _probe_in_flight = _probe_in_flight, False

        if was_probing:
            # The probe failed: back off further before the next one.
            _recovery_delay = min(LLM_SERVICE_RECOVERY_MAX,
                                  max(LLM_SERVICE_RECOVERY_SECONDS,
                                      _recovery_delay * 2))
            _recovery_at = time.monotonic() + _recovery_delay
            logger.warning(
                f"[llm_service] self-hosted probe failed — staying on the "
                f"fallback for another {_recovery_delay:.0f}s"
            )
        elif (not _degraded and fallback_available()
                and _consecutive_failures >= LLM_FALLBACK_AFTER_FAILURES):
            _degraded = True
            _recovery_delay = LLM_SERVICE_RECOVERY_SECONDS
            _recovery_at = time.monotonic() + _recovery_delay
            logger.warning(
                f"[llm_service] self-hosted LLM failed {_consecutive_failures}x — "
                f"using the OpenRouter fallback ({LLM_FALLBACK_MODEL}) and "
                f"re-probing in {_recovery_delay:.0f}s. NOTE: paid API."
            )
        return _degraded


def _mark_success(provider, usage=None):
    global _consecutive_failures, _degraded, _recovery_at, _recovery_delay
    global _probe_in_flight
    with _provider_lock:
        _call_counts[provider] = _call_counts.get(provider, 0) + 1
        if usage is not None:
            bucket = _token_counts[provider]
            bucket["prompt"] += getattr(usage, "prompt_tokens", None) or 0
            bucket["completion"] += getattr(usage, "completion_tokens", None) or 0
            cost = getattr(usage, "cost", None)
            if cost is not None and "cost" in bucket:
                bucket["cost"] += cost
        if provider == SELF_HOSTED:
            _consecutive_failures = 0
            _probe_in_flight = False
            if _degraded:
                # The probe answered: come straight back off the paid API.
                _degraded = False
                _recovery_at = 0.0
                _recovery_delay = 0.0
                logger.warning(
                    "[llm_service] self-hosted LLM is answering again — "
                    "returning to it and off the OpenRouter fallback"
                )


def force_fallback(reason=""):
    """
    Switch to OpenRouter immediately (used when the startup health check fails).

    Arms the same recovery cooldown as a mid-run failure: an llm-service that was
    merely restarting when the job started should be picked up minutes later, not
    written off for the whole run.
    """
    global _degraded, _recovery_at, _recovery_delay
    with _provider_lock:
        if not _degraded:
            _degraded = True
            _recovery_delay = LLM_SERVICE_RECOVERY_SECONDS
            _recovery_at = time.monotonic() + _recovery_delay
            logger.warning(
                f"[llm_service] using OpenRouter fallback ({LLM_FALLBACK_MODEL})"
                f"{f': {reason}' if reason else ''}. NOTE: paid API."
            )


def is_context_error(exc):
    """
    True when the server rejected the request for being too long, rather than
    for any other reason. The caller's response to this is to send LESS, not to
    retry the same payload — so it must be distinguishable from a transient
    error. Wordings differ per server (vLLM, TGI, llama.cpp, OpenRouter).
    """
    message = str(exc).lower()
    return any(
        phrase in message for phrase in (
            "maximum context length", "context length", "context_length_exceeded",
            "too many tokens", "input_tokens", "reduce the length",
            "longer than the maximum", "exceeds the maximum",
        )
    )


def discover_context_window():
    """
    Ask the server for its real context window (``GET /models`` →
    ``max_model_len``), falling back to LLM_SERVICE_CONTEXT_TOKENS when the
    server does not report one. vLLM exposes it; most others do not.
    """
    if not LLM_SERVICE_BASE_URL:
        return LLM_SERVICE_CONTEXT_TOKENS
    try:
        for model in get_llm_service_client().models.list().data:
            if model.id != LLM_SERVICE_MODEL:
                continue
            for attr in ("max_model_len", "context_length", "max_context_length"):
                value = getattr(model, attr, None)
                if value is None and hasattr(model, "model_extra"):
                    value = (model.model_extra or {}).get(attr)
                if value:
                    logger.info(f"[llm_service] discovered {attr}={value} for {model.id}")
                    return int(value)
    except Exception as e:
        logger.debug(f"[llm_service] context-window discovery failed: {e}")
    return LLM_SERVICE_CONTEXT_TOKENS


def _call_provider(provider, messages, temperature, max_tokens, json_mode, retries):
    """
    Run one provider's retry loop. Returns (text, usage).
    Raises LLMServiceError on exhaustion, or on a context-length rejection —
    which is never retried, because the same payload will fail identically.
    """
    if provider == SELF_HOSTED:
        client, model = get_llm_service_client(), LLM_SERVICE_MODEL
    else:
        client, model = get_fallback_client(), LLM_FALLBACK_MODEL

    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with _semaphore:
                response = client.chat.completions.create(**params)
            choice = response.choices[0]
            return (
                (choice.message.content or "").strip(),
                getattr(response, "usage", None),
                choice.finish_reason,
            )
        except openai.BadRequestError as e:
            # Not every server implements response_format; drop it once and retry.
            if params.get("response_format") and "response_format" in str(e):
                logger.info(
                    f"[llm_service] {provider} rejected response_format; retrying without it"
                )
                params.pop("response_format", None)
                continue
            raise LLMServiceError(f"{provider} rejected the request: {e}") from e
        except Exception as e:
            if is_context_error(e):
                # Retrying cannot help — surface it so the caller can split.
                raise LLMServiceError(f"{provider} context length exceeded: {e}") from e
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                delay = 3 * (2 ** (attempt - 1))
                logger.warning(
                    f"[llm_service] {provider} attempt {attempt}/{retries} failed "
                    f"({last_error}); retrying in {delay}s"
                )
                time.sleep(delay)

    raise LLMServiceError(f"{provider} failed after {retries} attempts: {last_error}")


def call_chat(messages, temperature=0.0, max_tokens=512, json_mode=False):
    """
    Call the LLM and return (text, provider, meta).

    Tries the self-hosted service first, then falls back to OpenRouter when it
    is unreachable or keeps failing. The fallback gets fewer retries, since by
    then the call has already waited out the self-hosted backoff.

    A context-length rejection is NOT failed over: sending the same oversized
    payload to a different provider does not make it fit, and (unlike a dead
    endpoint) it says nothing about the self-hosted service's health. It
    propagates so the caller can split the payload instead.
    """
    provider = current_provider()

    if provider == SELF_HOSTED:
        try:
            text, usage, finish_reason = _call_provider(
                provider, messages, temperature, max_tokens, json_mode, LLM_SERVICE_RETRIES)
            _mark_success(SELF_HOSTED, usage)
            return text, SELF_HOSTED, {"usage": usage, "finish_reason": finish_reason}
        except LLMServiceError as e:
            if is_context_error(e):
                raise
            if not fallback_available():
                raise
            _mark_failure()
            logger.warning(f"[llm_service] falling back to OpenRouter for this call: {e}")
        provider = OPENROUTER

    text, usage, finish_reason = _call_provider(
        provider, messages, temperature, max_tokens, json_mode,
        max(1, LLM_SERVICE_RETRIES - 1))
    _mark_success(OPENROUTER, usage)
    return text, OPENROUTER, {"usage": usage, "finish_reason": finish_reason}


def call_json(system, user, max_tokens=512, temperature=0.0, json_mode=True,
              parse_retries=1):
    """
    Call the model and parse the reply as a JSON object, using the pipeline's
    own sanitize -> parse -> repair cascade (processing/json_utils.py).

    Returns (obj_or_None, provider, meta) where meta carries `usage`,
    `finish_reason`, and `repaired` (True when the cascade had to fix the
    output — worth reporting, since a model needing frequent repair is a
    warning sign even when every call technically parses).
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_meta, last_provider = {}, current_provider()
    for attempt in range(parse_retries + 1):
        raw, provider, meta = call_chat(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        last_provider, last_meta = provider, meta
        if raw:
            try:
                obj = __import__("json").loads(raw)
                meta["repaired"] = False
                return obj, provider, meta
            except ValueError:
                pass
            try:
                obj = parse_llm_json(raw)
                meta["repaired"] = True
                return obj, provider, meta
            except JsonParseError:
                pass
        if attempt < parse_retries:
            logger.warning(
                f"[llm_service] unparseable reply ({(raw or '')[:60]!r}); "
                f"retrying ({attempt + 1}/{parse_retries})"
            )
    last_meta["repaired"] = True
    return None, last_provider, last_meta


def health_check():
    """
    Verify a provider answers before starting a long run.

    Probes the self-hosted service (``GET /models``, then a tiny chat call). If
    it is unreachable, switches the run to the OpenRouter fallback and probes
    that instead. Raises LLMServiceError only when neither can serve traffic.
    """
    if not LLM_SERVICE_BASE_URL:
        if not fallback_available():
            raise LLMServiceError(
                "LLM_SERVICE_BASE_URL is not set and no OpenRouter fallback is "
                "available (set OPEN_ROUTER_API_KEY or LLM_SERVICE_BASE_URL)."
            )
        force_fallback("LLM_SERVICE_BASE_URL is not set")
        return _probe_fallback()

    try:
        get_llm_service_client().models.list()
        logger.info(
            f"[llm_service] self-hosted LLM reachable at {LLM_SERVICE_BASE_URL} "
            f"(model={LLM_SERVICE_MODEL})"
        )
        return True
    except Exception as e:
        logger.info(
            f"[llm_service] /models probe failed ({type(e).__name__}: {e}); "
            f"trying a chat probe..."
        )

    try:
        _call_provider(SELF_HOSTED, [{"role": "user", "content": "ping"}], 0.0, 4, False, 1)
        _mark_success(SELF_HOSTED)
        logger.info(
            f"[llm_service] self-hosted LLM reachable at {LLM_SERVICE_BASE_URL} "
            f"(model={LLM_SERVICE_MODEL})"
        )
        return True
    except LLMServiceError as e:
        if not fallback_available():
            raise LLMServiceError(
                f"Self-hosted LLM unreachable and no OpenRouter fallback available: {e}"
            ) from e
        force_fallback(f"self-hosted LLM unreachable ({e})")
        return _probe_fallback()


def _probe_fallback():
    try:
        _call_provider(OPENROUTER, [{"role": "user", "content": "ping"}], 0.0, 4, False, 1)
        _mark_success(OPENROUTER)
        logger.info(f"[llm_service] OpenRouter fallback reachable (model={LLM_FALLBACK_MODEL})")
        return True
    except LLMServiceError as e:
        raise LLMServiceError(f"No LLM provider available — fallback also failed: {e}") from e
