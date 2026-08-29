"""
The golden dataset: fixed questions + ground-truth answers + the corpus
they should be answerable from. This is what makes regression testing
possible — same inputs, every run, forever (until you deliberately edit it).

Kept inline as Python for now. Once this grows past ~20 items, move it to
a JSON/YAML file loaded here instead — the rest of the pipeline doesn't care.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.target_app.rag_app import SimpleCorpus


@dataclass
class GoldenExample:
    question: str
    ground_truth: str


CORPUS_TEXTS = [
    "Token Optimizer is a Python/FastAPI middleware library with 15+ modules "
    "and 33 REST endpoints for reducing LLM token usage in production apps.",
    "The context router in Token Optimizer decides which retrieved chunks "
    "are sent to the LLM, using a cross-encoder reranker built on "
    "sentence-transformers to score relevance.",
    "Token Optimizer supports Gemini as the primary LLM provider and "
    "Anthropic as a secondary provider, with cache_control annotations "
    "for Anthropic's prompt caching feature.",
    "The telemetry module in Token Optimizer tracks token counts, latency, "
    "and cost per request across all configured LLM providers.",
    "Token Optimizer's roadmap has P0 items partially complete, with the "
    "context router's signal wiring still an open item.",
]

GOLDEN_SET: list[GoldenExample] = [
    GoldenExample(
        question="What is Token Optimizer?",
        ground_truth=(
            "A Python/FastAPI middleware library for reducing LLM token "
            "usage, with 15+ modules and 33 REST endpoints."
        ),
    ),
    GoldenExample(
        question="How does the context router decide which chunks to send?",
        ground_truth=(
            "It uses a cross-encoder reranker built on sentence-transformers "
            "to score chunk relevance before sending them to the LLM."
        ),
    ),
    GoldenExample(
        question="Which LLM providers does Token Optimizer support?",
        ground_truth="Gemini as primary, Anthropic as secondary.",
    ),
    GoldenExample(
        question="What does the telemetry module track?",
        ground_truth="Token counts, latency, and cost per request.",
    ),
    GoldenExample(
        question="What is still open in the roadmap?",
        ground_truth="The context router's signal wiring is still open.",
    ),
]


def get_corpus() -> SimpleCorpus:   return SimpleCorpus.from_texts(CORPUS_TEXTS)
