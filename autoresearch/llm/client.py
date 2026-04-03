"""Provider-aware LLM client with graceful local fallback."""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from ..config.schema import LLMConfig


@dataclass(slots=True)
class LLMClient:
    """Simple synchronous LLM client used by agents."""

    config: LLMConfig

    def prompt(self, prompt: str) -> str:
        """Get a completion for a prompt.

        Falls back to deterministic local stub when provider cannot be used
        (e.g., missing API key or unsupported provider) so offline tests stay stable.
        """
        provider = (self.config.provider or "").strip().lower()
        if provider == "openai":
            return self._prompt_openai(prompt)
        return self._fallback(prompt)

    def _prompt_openai(self, prompt: str) -> str:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            return self._fallback(prompt)

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError):
            return self._fallback(prompt)

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            return self._fallback(prompt)

    @staticmethod
    def _fallback(prompt: str) -> str:
        return f"[LLM response to: {prompt[:80]}...]"
