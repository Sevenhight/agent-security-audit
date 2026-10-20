"""Thin DeepSeek client over the OpenAI-compatible chat API.

Only stdlib is used (``urllib``) so the repository runs with a bare Python
install plus a key. Swap ``DeepSeekClient`` for any OpenAI-compatible client
if you prefer; the agent only depends on :meth:`chat`.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")


class LLMError(RuntimeError):
    """Raised when the upstream API cannot produce a usable answer."""


class DeepSeekClient:
    """Minimal chat-completion client.

    Parameters mirror the OpenAI schema so switching providers is a matter of
    changing ``base_url`` and ``model``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise LLMError("DEEPSEEK_API_KEY is not set")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                # Retry only transient server-side failures.
                if exc.code >= 500 and attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(f"HTTP {exc.code}: {detail}") from exc
            except (KeyError, json.JSONDecodeError) as exc:
                raise LLMError(f"unexpected response shape: {exc}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(f"network error: {exc}") from exc

        raise LLMError("retries exhausted")


class MockClient:
    """Deterministic stand-in for offline runs, tests and CI.

    It replays a scripted list of assistant messages so the whole pipeline can
    be exercised without a key. This is also what makes the repository
    reviewable by someone who does not want to spend credits.
    """

    def __init__(self, scripted: list[str]) -> None:
        self._scripted = list(scripted)
        self._cursor = 0
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self._cursor >= len(self._scripted):
            return '{"type": "final", "answer": "[mock] script exhausted"}'
        reply = self._scripted[self._cursor]
        self._cursor += 1
        return reply
