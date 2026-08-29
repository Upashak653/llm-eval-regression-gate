"""
The actual "gate" — decides pass/fail. Deliberately separate from
metrics.py: scoring is "what did we measure", gating is "is that OK".

Two kinds of checks:
1. Absolute floor — a metric must never drop below this, regardless of baseline.
2. Regression tolerance — a metric must not drop more than X vs the last
   committed baseline (catches slow decay even if still above the floor).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parents[2] / "baselines" / "baseline_scores.json"

# Tune these once you have real baseline numbers from your local model.
ABSOLUTE_FLOORS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.60,
}

# Max allowed drop vs baseline, in absolute score points (e.g. 0.05 = 5%).
REGRESSION_TOLERANCE = 0.05


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    scores: dict
    baseline: dict


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text())


def save_baseline(scores: dict) -> None:
    BASELINE_PATH.write_text(json.dumps(scores, indent=2))


def check_gate(scores: dict) -> GateResult:
    baseline = load_baseline()
    failures = []

    for metric, value in scores.items():
        floor = ABSOLUTE_FLOORS.get(metric)
        if floor is not None and value < floor:
            failures.append(
                f"{metric}={value:.3f} is below absolute floor {floor:.3f}"
            )

        baseline_value = baseline.get(metric)
        if baseline_value is not None:
            drop = baseline_value - value
            if drop > REGRESSION_TOLERANCE:
                failures.append(
                    f"{metric}={value:.3f} dropped {drop:.3f} vs baseline "
                    f"{baseline_value:.3f} (tolerance {REGRESSION_TOLERANCE:.3f})"
                )

    return GateResult(passed=not failures, failures=failures, scores=scores, baseline=baseline)
