"""
Optional FastAPI wrapper around the eval runner — lets you trigger an eval
run over HTTP (e.g. from a local dashboard, or manually via curl) without
going through pytest. Not used by CI; CI calls pytest directly.
"""
from __future__ import annotations

from fastapi import FastAPI

from app.eval.runner import run_eval
from app.gate.thresholds import check_gate, save_baseline

app = FastAPI(title="LLM Eval Gate")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate():
    result = run_eval()
    gate = check_gate(result["scores"])
    return {
        "passed": gate.passed,
        "failures": gate.failures,
        "scores": gate.scores,
        "baseline": gate.baseline,
    }


@app.post("/baseline/update")
def update_baseline():
    """
    Deliberately separate from /evaluate — updating the baseline should be
    a conscious action (e.g. after you've reviewed and accepted a change),
    never an automatic side effect of running the eval.
    """
    result = run_eval()
    save_baseline(result["scores"])
    return {"saved": result["scores"]}
