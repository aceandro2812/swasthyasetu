"""
llm_router.py — Multi-Provider LLM Router for SwasthyaSetu / MedSarathi

Distributes LLM calls across multiple free-tier providers (Gemini, OpenRouter)
using round-robin selection with automatic failover on 429 rate-limit errors.

Setup:
  GOOGLE_API_KEY    — Gemini free tier (required, already in .env)
  OPENROUTER_API_KEY — OpenRouter free models (optional but strongly recommended)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, List, Optional

logger = logging.getLogger("swasthyasetu.llm_router")

# ---------------------------------------------------------------------------
# Provider cooldown window (seconds) — how long to skip a provider after a 429
# ---------------------------------------------------------------------------
DEFAULT_COOLDOWN_SECONDS = 60  # 1 minute cooldown after a 429


# ---------------------------------------------------------------------------
# Provider dataclass
# ---------------------------------------------------------------------------
@dataclass
class LLMProvider:
    name: str                          # Human-readable label (for logs)
    model_id: str                      # Model identifier (passed to the call fn)
    call_fn: Callable[[str], str]      # fn(prompt) -> text
    # Internal state — managed by LLMRouter
    _cooldown_until: float = field(default=0.0, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def is_available(self) -> bool:
        return time.time() >= self._cooldown_until

    def mark_rate_limited(self, cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS) -> None:
        with self._lock:
            self._cooldown_until = time.time() + cooldown_seconds
        logger.warning(
            f"[Router] Provider '{self.name}' rate-limited. "
            f"Cooling down for {cooldown_seconds}s."
        )

    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.time())


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
class LLMRouter:
    """
    Round-robin LLM router with automatic failover on 429 errors.

    Usage:
        router = LLMRouter()
        text = router.generate_content("Your prompt here")
    """

    def __init__(self) -> None:
        self._providers: List[LLMProvider] = []
        self._index: int = 0
        self._lock: Lock = Lock()
        self._build_providers()

    # ------------------------------------------------------------------
    # Provider construction
    # ------------------------------------------------------------------
    def _build_providers(self) -> None:
        """Instantiate all configured free-tier providers."""

        # --- 1. Google Gemini (primary, existing) ---
        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if google_api_key:
            try:
                from google import genai
                from google.genai import types as genai_types

                _gemini_client = genai.Client(api_key=google_api_key)
                _gemini_model = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

                def _call_gemini(prompt: str) -> str:
                    response = _gemini_client.models.generate_content(
                        model=_gemini_model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )
                    if response.text:
                        return response.text
                    raise ValueError("Empty response from Gemini.")

                self._providers.append(
                    LLMProvider(
                        name=f"Gemini ({_gemini_model})",
                        model_id=_gemini_model,
                        call_fn=_call_gemini,
                    )
                )
                logger.info(f"[Router] Registered provider: Gemini ({_gemini_model})")
            except Exception as e:
                logger.error(f"[Router] Failed to initialize Gemini provider: {e}")

        # --- 2. OpenRouter free models ---
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if openrouter_key:
            try:
                from openai import OpenAI, RateLimitError

                _or_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                )

                # List of free OpenRouter models — ordered by capability preference
                _free_models = [
                    ("meta-llama/llama-4-maverick:free",              "Llama 4 Maverick"),
                    ("deepseek/deepseek-chat-v3-0324:free",           "DeepSeek Chat V3"),
                    ("google/gemma-3-27b-it:free",                    "Gemma 3 27B"),
                    ("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral Small 3.1"),
                    ("meta-llama/llama-3.3-70b-instruct:free",        "Llama 3.3 70B"),
                    ("nvidia/llama-3.1-nemotron-nano-8b-v1:free",     "Nemotron Nano 8B"),
                ]

                for model_id, label in _free_models:
                    # Capture model_id in closure
                    def _make_call_fn(mid: str) -> Callable[[str], str]:
                        def _call_openrouter(prompt: str) -> str:
                            completion = _or_client.chat.completions.create(
                                model=mid,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.2,
                                extra_headers={
                                    "HTTP-Referer": "https://medsarathi.onrender.com",
                                    "X-Title": "MedSarathi Health AI",
                                },
                            )
                            text = completion.choices[0].message.content
                            if text:
                                return text
                            raise ValueError(f"Empty response from OpenRouter ({mid}).")
                        return _call_openrouter

                    self._providers.append(
                        LLMProvider(
                            name=f"OpenRouter/{label}",
                            model_id=model_id,
                            call_fn=_make_call_fn(model_id),
                        )
                    )
                    logger.info(f"[Router] Registered provider: OpenRouter/{label}")

            except ImportError:
                logger.error(
                    "[Router] 'openai' package not installed. "
                    "Run: pip install openai  (needed for OpenRouter)"
                )
            except Exception as e:
                logger.error(f"[Router] Failed to initialize OpenRouter providers: {e}")

        if not self._providers:
            logger.error(
                "[Router] No LLM providers available! "
                "Set GOOGLE_API_KEY and/or OPENROUTER_API_KEY in .env"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Return True if at least one provider is registered."""
        return bool(self._providers)

    def generate_content(self, prompt: str, max_attempts: Optional[int] = None) -> str:
        """
        Send `prompt` to the next available LLM provider.

        Cycles through providers round-robin. On a 429 / rate-limit error
        the offending provider is marked as cooling down and the next provider
        is tried automatically.  If all providers are cooling down the call
        waits for the shortest cooldown and retries.

        Returns the LLM response text or an "Error: ..." string on total failure.
        """
        if not self._providers:
            return "Error: No LLM providers configured."

        n = len(self._providers)
        if max_attempts is None:
            max_attempts = n * 3  # try each provider up to 3 times before giving up

        attempts = 0
        while attempts < max_attempts:
            provider = self._pick_provider()

            if provider is None:
                # All providers cooling down — wait for the shortest cooldown
                min_wait = min(p.cooldown_remaining() for p in self._providers)
                logger.warning(
                    f"[Router] All providers cooling down. "
                    f"Waiting {min_wait:.1f}s for next available..."
                )
                time.sleep(min_wait + 0.5)
                attempts += 1
                continue

            try:
                logger.info(
                    f"[Router] Calling provider '{provider.name}' "
                    f"(attempt {attempts + 1}/{max_attempts})"
                )
                start = time.time()
                result = provider.call_fn(prompt)
                duration = time.time() - start
                logger.info(
                    f"[Router] Provider '{provider.name}' responded in {duration:.2f}s"
                )
                return result

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(
                    kw in err_str
                    for kw in ("429", "rate_limit", "resource_exhausted", "quota", "too many")
                )

                if is_rate_limit:
                    # Try to parse retry-after from the error message
                    cooldown = DEFAULT_COOLDOWN_SECONDS
                    import re
                    m = re.search(r"retry.{0,15}?(\d+)\s*s", err_str)
                    if m:
                        cooldown = min(int(m.group(1)) + 5, 300)
                    provider.mark_rate_limited(cooldown)
                    logger.warning(
                        f"[Router] Provider '{provider.name}' returned 429. "
                        "Failing over to next provider."
                    )
                else:
                    # Non-rate-limit error — log and try next provider once
                    logger.error(
                        f"[Router] Provider '{provider.name}' failed with non-rate-limit error: {e}"
                    )
                    # Brief pause before trying next
                    time.sleep(1)

                attempts += 1

        logger.error("[Router] All providers exhausted. Returning error.")
        return "Error: All LLM providers are currently rate-limited or unavailable. Please try again in a minute."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pick_provider(self) -> Optional[LLMProvider]:
        """
        Return the next available provider in round-robin order,
        or None if all providers are currently cooling down.
        """
        with self._lock:
            n = len(self._providers)
            for _ in range(n):
                candidate = self._providers[self._index % n]
                self._index = (self._index + 1) % n
                if candidate.is_available():
                    return candidate
        return None

    def status(self) -> List[dict]:
        """Return a dict summary of all provider states (useful for /health endpoint)."""
        return [
            {
                "name": p.name,
                "model": p.model_id,
                "available": p.is_available(),
                "cooldown_remaining_s": round(p.cooldown_remaining(), 1),
            }
            for p in self._providers
        ]
