"""
Ties dataset + target_app + metrics together into one call: run_eval().
This is what both pytest and the FastAPI /evaluate endpoint call.
"""
from __future__ import annotations

from app.eval.dataset import GOLDEN_SET, get_corpus
from app.eval.metrics import score
from app.target_app.llm_client import get_llm_client
from app.target_app.rag_app import RagPipeline


def _build_ragas_judge():
    """
    RAGAS wants an LLM + embeddings for its own judging (separate from the
    target app's LLM, though here both point at the same local Ollama
    instance). Wrapped via langchain so RAGAS's LangchainLLMWrapper accepts
    it. Import is local so the rest of the module works even if langchain
    isn't installed (e.g. mock/dry-run mode).
    """
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        from app.target_app.llm_client import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL

        judge_llm = LangchainLLMWrapper(
            ChatOllama(base_url=OLLAMA_BASE_URL, model=CHAT_MODEL)
        )
        judge_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)
        )
        return judge_llm, judge_embeddings
    except Exception as e:
        import sys

        print(f"[runner] Could not build local RAGAS judge ({e}); "
              "scoring will use RAGAS defaults (needs OPENAI_API_KEY) "
              "or fail loudly.", file=sys.stderr)
        return None, None


def run_eval() -> dict:
    """
    Runs the full golden set through the RAG pipeline, then scores it.
    Returns {"scores": {...}, "raw_results": [...]} — raw_results kept for
    debugging/inspection, scores is what the gate checks.
    """
    llm = get_llm_client()
    corpus = get_corpus()
    pipeline = RagPipeline(corpus=corpus, llm=llm)

    results = [pipeline.answer(ex.question) for ex in GOLDEN_SET]
    ground_truths = [ex.ground_truth for ex in GOLDEN_SET]

    judge_llm, judge_embeddings = _build_ragas_judge()
    scores = score(results, ground_truths, llm=judge_llm, embeddings=judge_embeddings)

    return {"scores": scores, "raw_results": results}


if __name__ == "__main__":
    import json

    out = run_eval()
    print(json.dumps(out["scores"], indent=2))

    import os
    if os.getenv("EVAL_DEBUG"):
        from app.eval.metrics import build_ragas_dataset
        print("\n--- raw results (per example) ---")
        for r in out["raw_results"]:
            print(json.dumps(r, indent=2))