"""
Thin LLM client so the rest of the app never talks to Ollama's HTTP API directly.

Why this exists: your target app AND RAGAS's judge both need "call an LLM"
and "call an embedder". If those calls are scattered, swapping models later
(or mocking them in CI where Ollama isn't reachable) means hunting through
every file. One seam here = one place to change.
"""
from __future__ import annotations

import os
import httpx


try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "qwen2.5-coder:14b")
EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "nomic-embed-text")


print(f"[llm_client] Using CHAT_MODEL={CHAT_MODEL!r} EMBED_MODEL={EMBED_MODEL!r} "
      f"OLLAMA_BASE_URL={OLLAMA_BASE_URL!r}")

class OllamaClient:
    """Real client — used on your machine where Ollama is running."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = 120.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def chat(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.post(
            "/api/chat",
            json={"model": CHAT_MODEL, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def embed(self, text: str) -> list[float]:
        resp = self._client.post(
            "/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def is_reachable(self) -> bool:
        try:
            r = self._client.get("/api/tags", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False


class MockClient:
    """
    Deterministic stand-in used when Ollama isn't reachable (e.g. this sandbox,
    or a CI runner with no local model). Lets you validate pipeline WIRING
    without validating actual answer QUALITY — RAGAS scores from this will be
    meaningless, but pytest collection / gate logic / CI plumbing can still be
    proven correct.
    """

    def chat(self, prompt: str, system: str | None = None) -> str:
        return f"[mock answer for prompt of length {len(prompt)}]"

    def embed(self, text: str) -> list[float]:
        # Cheap deterministic pseudo-embedding: hash-based, fixed dimension.
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:32]]

    def is_reachable(self) -> bool:
        return True


def get_llm_client() -> OllamaClient | MockClient:
    """
    Auto-selects: real Ollama if reachable, otherwise falls back to the mock
    with a loud stderr warning. This is the ONLY place that decision is made.
    """
    real = OllamaClient()
    if real.is_reachable():
        return real

    import sys

    print(
        f"[llm_client] WARNING: Ollama not reachable at {OLLAMA_BASE_URL}. "
        "Falling back to MockClient — eval scores will NOT be meaningful.",
        file=sys.stderr,
    )
    return MockClient()
