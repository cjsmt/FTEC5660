# =====================================================
#  Evaluation code
# =====================================================

import main


def evaluate(scores, groundtruth, threshold=0.5):
    """
    scores: list of floats in [0, 1], length = 5
    groundtruth: list of ints (0 or 1), length = 5
    """
    assert len(scores) == 5
    assert len(groundtruth) == 5

    correct = 0
    decisions = []

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
        "final_score": final_score
    }


if __name__ == "__main__":
    scores = []
    test_cases = ["CV_1.pdf", "CV_2.pdf", "CV_3.pdf", "CV_4.pdf", "CV_5.pdf"]
    for test_case in test_cases:
        result = main.run_pipeline("datasets/" + test_case)
        scores.append(result.get("report", {}).get("average_score"))
    groundtruth = [1, 1, 1, 0, 0] # Do not modify
    result = evaluate(scores, groundtruth)
    print(result)
