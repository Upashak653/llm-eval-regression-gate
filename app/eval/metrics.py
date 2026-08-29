"""
Wraps RAGAS so nothing else in the codebase imports it directly. RAGAS's
API has churned across versions; if it changes again, this is the only
file that needs to.

RAGAS needs an LLM to act as judge (for faithfulness, answer relevancy)
and an embedder (for context precision/recall). We point both at your
local Ollama model via a LangChain-compatible wrapper, so scoring itself
costs zero API dollars too — consistent with the rest of this project.
"""
from __future__ import annotations

import sys
import types


def _patch_ragas_vertexai_import() -> None:
    """
    ragas/llms/base.py unconditionally does:
        from langchain_community.chat_models.vertexai import ChatVertexAI
        from langchain_community.llms import VertexAI
    That submodule was fully removed from recent langchain_community
    releases (deprecated since 0.0.12, physically gone in newer versions),
    which crashes `import ragas` for EVERYONE, not just VertexAI users —
    see https://github.com/explodinggradients/ragas/issues/2753.

    We never use VertexAI, so we pre-register harmless stub modules in
    sys.modules before ragas imports them. If the real submodule already
    exists (older langchain_community), this does nothing — real thing wins.
    """
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        try:
            import langchain_community.chat_models.vertexai  # noqa: F401
        except ImportError:
            stub = types.ModuleType("langchain_community.chat_models.vertexai")
            stub.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules["langchain_community.chat_models.vertexai"] = stub

    if "langchain_community.llms" in sys.modules:
        llms_mod = sys.modules["langchain_community.llms"]
        if not hasattr(llms_mod, "VertexAI"):
            llms_mod.VertexAI = type("VertexAI", (), {})
    else:
        try:
            import langchain_community.llms as llms_mod  # noqa: F401

            if not hasattr(llms_mod, "VertexAI"):   llms_mod.VertexAI = type("VertexAI", (), {})
        except ImportError:
            pass


_patch_ragas_vertexai_import()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def build_ragas_dataset(results: list[dict], ground_truths: list[str]) -> Dataset:
    """
    results: list of {"question", "answer", "contexts"} from RagPipeline.answer()
    ground_truths: parallel list of reference answers from the golden set
    """
    return Dataset.from_dict(
        {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": ground_truths,
        }
    )


def score(results: list[dict], ground_truths: list[str], llm=None, embeddings=None) -> dict:
    """
    Returns a flat dict: {"faithfulness": 0.83, "answer_relevancy": 0.91, ...}
    averaged across the whole golden set.

    llm/embeddings: optional RAGAS-compatible wrappers around your local
    Ollama model. If omitted, RAGAS defaults to OpenAI — which will fail
    without an API key, by design, so you notice and wire in the local
    judge explicitly (see runner.py).
    """
    ds = build_ragas_dataset(results, ground_truths)
    kwargs = {}
    if llm is not None: kwargs["llm"] = llm
    if embeddings is not None:  kwargs["embeddings"] = embeddings

    import os
    if os.getenv("EVAL_DEBUG"):
        kwargs["raise_exceptions"] = True

    report = evaluate(ds, metrics=METRICS, **kwargs)

    if hasattr(report, "items"):    return {k: float(v) for k, v in report.items()}
    df = report.to_pandas()
    metric_names = [m.name for m in METRICS]
    return {name: float(df[name].mean()) for name in metric_names if name in df.columns}