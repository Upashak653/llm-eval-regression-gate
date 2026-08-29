"""
This is the test CI runs. It's deliberately thin: run_eval() and check_gate()
hold all the logic, so this file stays readable as the single source of
truth for "what does a passing build mean".

To PROVE the gate works (for your README demo), do one of:
  a) Edit app/target_app/rag_app.py's system prompt to something worse
     (e.g. remove "using ONLY the provided context") and rerun -> watch
     faithfulness drop and this test fail.
  b) Temporarily lower baselines/baseline_scores.json values way down,
     confirm the test passes, then restore -> shows the tolerance logic works.
  c) Point LLM_CHAT_MODEL at a weaker/different local model and rerun.
"""
import pytest

from app.eval.runner import run_eval
from app.gate.thresholds import check_gate


@pytest.fixture(scope="module")
def eval_result():
    return run_eval()


def test_eval_gate_passes(eval_result):
    result = check_gate(eval_result["scores"])

    if not result.passed:
        details = "\n  - ".join(result.failures)
        pytest.fail(
            f"LLM eval regression gate FAILED:\n  - {details}\n\n"
            f"Current scores: {result.scores}\nBaseline: {result.baseline}"
        )


def test_all_expected_metrics_present(eval_result):
    expected = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    assert expected.issubset(eval_result["scores"].keys())
