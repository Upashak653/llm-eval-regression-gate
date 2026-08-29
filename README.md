# LLM Eval Regression Gate

A CI/CD gate that catches silent regressions in LLM applications — when a
prompt edit, chunking change, or model swap quietly makes answers less
faithful or less relevant, this fails the build instead of letting it ship.

Built as a regression harness around [Token Optimizer](#), tested with
[RAGAS](https://github.com/explodinggradients/ragas), running entirely on a
**local model via Ollama** (Qwen2.5-Coder 14B) — zero API cost to run evals.

## Why this exists

Traditional tests check "does the code run." They don't check "did the
*answers* get worse." A one-line prompt edit can silently tank faithfulness
or context recall with no exception thrown, no red X in CI — until a user
notices the bot is now confidently wrong. This project makes that failure
mode visible and blocking.

## Architecture

```
app/
├── target_app/        # the RAG pipeline being regression-tested
│   ├── llm_client.py   # local Ollama client (+ mock fallback for CI dry-runs)
│   └── rag_app.py       # retrieve -> rerank -> answer
├── eval/
│   ├── dataset.py       # fixed golden Q&A set
│   ├── metrics.py       # RAGAS metric wrapper
│   └── runner.py         # ties it all together -> run_eval()
├── gate/
│   └── thresholds.py    # pass/fail logic: absolute floors + regression tolerance
└── api/
    └── server.py         # optional FastAPI /evaluate endpoint

tests/test_regression.py  # the pytest CI actually runs
baselines/baseline_scores.json  # committed "last known good" scores
.github/workflows/eval-gate.yml # runs the gate on PRs touching prompts/eval
```

## Metrics tracked (via RAGAS)

- **Faithfulness** — is the answer actually supported by retrieved context?
- **Answer relevancy** — does the answer address the question asked?
- **Context precision** — are retrieved chunks actually relevant?
- **Context recall** — did retrieval find what was needed to answer?

## Setup

```bash
pip install -r requirements.txt

# Requires Ollama running locally with a chat + embedding model pulled:
ollama pull qwen2.5-coder:14b
ollama pull nomic-embed-text
```

## Running the gate

```bash
# One-off eval run, prints raw scores
python -m app.eval.runner

# The actual CI gate
pytest tests/test_regression.py -v

# Local API
uvicorn app.api.server:app --reload
```

## Demo: proving the gate catches a regression

## Demo: proving the gate catches a regression

This is the core recruiter-facing demo — a deliberate retrieval regression
that the gate catches. Note: a prompt-only regression (e.g. removing the
grounding instruction) was tried first and did NOT reliably break scores —
capable models often stay grounded even without being told to. A retrieval
bug is more realistic and deterministic, so that's the demo below.

1. Run the baseline eval and confirm it passes:
```bash
   pytest tests/test_regression.py -v
```
2. Open `app/target_app/rag_app.py`, find `retrieve()`, and flip the sort
   direction — this simulates a reranker sort-order bug, retrieving the
   LEAST relevant chunks instead of the most relevant ones:
```python
   # before
   scored.sort(key=lambda pair: pair[0], reverse=True)
   # after (regression)
   scored.sort(key=lambda pair: pair[0], reverse=False)
```
3. Re-run the gate:
```bash
   pytest tests/test_regression.py -v
```
   All four metrics crater at once — the pipeline is now feeding the LLM
   irrelevant context for every question:
## Updating the baseline (intentional changes)

If a change is a deliberate improvement (not a regression), update the
committed baseline explicitly rather than letting the gate silently adapt:

```bash
curl -X POST http://localhost:8000/baseline/update
```

## Notes on CI

GitHub-hosted runners can't reach a local Ollama instance. To run this gate
in real CI, either:
- use a **self-hosted runner** on a machine with Ollama already running, or
- install + pull a small model directly in the workflow (see comments in
  `.github/workflows/eval-gate.yml`).

Without either, CI falls back to a deterministic mock LLM — enough to prove
the pipeline wiring and gate logic are correct, but the scores themselves
are meaningless. This is intentional and logged loudly rather than silently
passing.
