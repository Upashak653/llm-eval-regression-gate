"""
Minimal RAG pipeline standing in for Token Optimizer's context router +
cross-encoder reranker. This is the "app under test" — the thing whose
prompts/chunking/model you'll deliberately change to prove the CI gate
catches regressions.

Kept dependency-light on purpose: sentence-transformers for embeddings/
reranking (already in your stack), local LLM via llm_client for generation.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.target_app.llm_client import get_llm_client


@dataclass
class Chunk:
    id: str
    text: str


class SimpleCorpus:
    """In-memory corpus — swap for pgvector/FTS5 later; interface stays the same."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    @classmethod
    def from_texts(cls, texts: list[str]) -> "SimpleCorpus":
        return cls([Chunk(id=f"c{i}", text=t) for i, t in enumerate(texts)])


class RagPipeline:
    def __init__(self, corpus: SimpleCorpus, top_k: int = 3, llm=None):
        self.corpus = corpus
        self.top_k = top_k
        self.llm = llm or get_llm_client()
        self._embedder = None  # lazy-loaded, see _embed()

    def _embed(self, text: str) -> list[float]:
        # Prefer sentence-transformers locally if available (matches your
        # stack); fall back to the llm_client's embed() so this still works
        # against Ollama-served embedding models with zero extra deps.
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedder = False  # sentinel: unavailable, use llm.embed

        if self._embedder:
            return self._embedder.encode(text).tolist()
        return self.llm.embed(text)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-8
        nb = math.sqrt(sum(y * y for y in b)) or 1e-8
        return dot / (na * nb)

    def retrieve(self, question: str) -> list[Chunk]:
        q_emb = self._embed(question)
        scored = [
            (self._cosine(q_emb, self._embed(c.text)), c) for c in self.corpus.chunks
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [c for _, c in scored[: self.top_k]]

    def answer(self, question: str) -> dict:
        contexts = self.retrieve(question)
        context_block = "\n\n".join(c.text for c in contexts)

        # THIS PROMPT is exactly the kind of thing you'd deliberately change
        # to demonstrate a regression later (tone shift, instruction removed,
        # format change, etc).
        system = (
        "Answer the user's question using ONLY the provided context. "
        "If the context doesn't contain the answer, say you don't know."
        )
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        response = self.llm.chat(prompt, system=system)
        return {
            "question": question,
            "answer": response,
            "contexts": [c.text for c in contexts],
        }
