"""
Simple evaluation script.

It runs the full CV verification pipeline on 5 test resumes
(`datasets/CV_1.pdf` ... `datasets/CV_5.pdf`), collects the `average_score`
for each resume, compares them against a binary groundtruth label
([1, 1, 1, 0, 0]) using a fixed threshold to compute accuracy,
and saves all 5 reports plus evaluation result to a single JSON file in output/.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import main


def evaluate(scores: List[float], groundtruth: List[int], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate average_scores against binary groundtruth labels.

    - scores: list of floats in [0, 1], length = 5
    - groundtruth: list of ints (0 or 1), length = 5
    - threshold: score > threshold → predict 1, else 0
    """
    assert len(scores) == 5
    assert len(groundtruth) == 5

    correct = 0
    decisions: List[int] = []

    for s, gt in zip(scores, groundtruth):
        pred = 1 if s > threshold else 0
        decisions.append(pred)
        if pred == gt:
            correct += 1

    final_score = correct / len(scores)

    return {
        "decisions": decisions,
        "correct": correct,
        "total": len(scores),
        "final_score": final_score,
        "threshold": threshold,
    }


if __name__ == "__main__":
    test_cases = ["CV_1.pdf", "CV_2.pdf", "CV_3.pdf", "CV_4.pdf", "CV_5.pdf"]
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"

    print("=== Running CV verification pipeline on test cases ===")
    scores: List[float] = []
    reports: List[Dict[str, Any]] = []
    for i, filename in enumerate(test_cases, start=1):
        pdf_path = base_dir / "datasets" / filename
        print(f"[{i}/{len(test_cases)}] Processing {pdf_path} ...")
        state = main.run_pipeline(str(pdf_path))
        report = state.get("report") or {}
        reports.append({"filename": filename, "report": report})
        avg_score = report.get("average_score")
        scores.append(float(avg_score) if avg_score is not None else 0.0)
        print(f"    average_score = {avg_score}")

    groundtruth = [1, 1, 1, 0, 0]  # Do not modify
    print("\n=== Evaluating against groundtruth ===")
    eval_result = evaluate(scores, groundtruth)
    print(f"Scores:      {scores}")
    print(f"Decisions:   {eval_result['decisions']}")
    print(f"Groundtruth: {groundtruth}")
    print(f"Correct: {eval_result['correct']} / {eval_result['total']}, final_score = {eval_result['final_score']:.3f}")

    # Save all 5 reports and evaluation result to a single JSON file
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{timestamp}.json"
    combined = {
        "reports": reports,
        "scores": scores,
        "evaluation": eval_result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {out_path}")

