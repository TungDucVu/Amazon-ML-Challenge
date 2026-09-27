import sys
from typing import Dict, List, Set, Union
import numpy as np
import pandas as pd


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    
    - True empty & Pred empty -> 1.0
    - True empty & Pred non-empty -> 0.0
    - True non-empty & Pred empty -> 0.0
    """
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    tp = len(true_ids.intersection(pred_ids))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return float(f05)


def parse_id_list(id_str: str) -> Set[str]:
    """Parses a comma-separated ID string into a set of non-empty stripped IDs."""
    if not isinstance(id_str, str) or not id_str.strip():
        return set()
    return {x.strip() for x in id_str.split(",") if x.strip()}


def compute_macro_f05(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    s1_col: str = "source1_entity_id",
    gt_match_col: str = "matched_entity_ids",
    pred_match_col: str = "matched_entity_ids"
) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5 score across all Source 1 entities in ground truth.
    
    - Computes F_0.5 per S1 entity first.
    - Takes arithmetic mean across all S1 entities.
    """
    gt_map = {
        row[s1_col]: parse_id_list(row[gt_match_col])
        for _, row in ground_truth_df.iterrows()
    }
    
    pred_map = {
        row[s1_col]: parse_id_list(row[pred_match_col])
        for _, row in predictions_df.iterrows()
    }

    scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id, true_ids in gt_map.items():
        pred_ids = pred_map.get(s1_id, set())
        score = compute_entity_f05(true_ids, pred_ids)
        scores.append(score)

        if len(true_ids) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    singleton_f05 = float(np.mean(singleton_scores)) if singleton_scores else 0.0
    non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_f05": singleton_f05,
        "non_singleton_f05": non_singleton_f05,
        "total_entities": len(scores),
        "singleton_count": len(singleton_scores),
        "non_singleton_count": len(non_singleton_scores),
    }


def run_unit_tests():
    """Runs standard test suite (Tests A-F) with exact floating-point assertions."""
    print("Running metric unit test suite...")

    # Test A: True empty, Pred empty -> 1.0
    score_a = compute_entity_f05(set(), set())
    assert abs(score_a - 1.0) < 1e-6, f"Test A failed: expected 1.0, got {score_a}"
    print("  [PASS] Test A (True empty, Pred empty): 1.0")

    # Test B: True empty, Pred non-empty -> 0.0
    score_b = compute_entity_f05(set(), {"S2-001"})
    assert abs(score_b - 0.0) < 1e-6, f"Test B failed: expected 0.0, got {score_b}"
    print("  [PASS] Test B (True empty, Pred non-empty): 0.0")

    # Test C: True non-empty, Pred empty -> 0.0
    score_c = compute_entity_f05({"S2-001"}, set())
    assert abs(score_c - 0.0) < 1e-6, f"Test C failed: expected 0.0, got {score_c}"
    print("  [PASS] Test C (True non-empty, Pred empty): 0.0")

    # Test D: Exact Match 2 elements -> 1.0
    score_d = compute_entity_f05({"S2-001", "S3-002"}, {"S2-001", "S3-002"})
    assert abs(score_d - 1.0) < 1e-6, f"Test D failed: expected 1.0, got {score_d}"
    print("  [PASS] Test D (Exact match): 1.0")

    # Test E: True 2 elems, Pred 1 correct (High Precision: P=1.0, R=0.5) -> F0.5 = 5/6 = 0.8333333...
    score_e = compute_entity_f05({"S2-001", "S3-002"}, {"S2-001"})
    expected_e = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)  # 5/6 = 0.833333...
    assert abs(score_e - expected_e) < 1e-6, f"Test E failed: expected {expected_e}, got {score_e}"
    print(f"  [PASS] Test E (P=1.0, R=0.5): {score_e:.4f} (expected {expected_e:.4f})")

    # Test F: True 1 elem, Pred 2 elems (1 correct, 1 wrong: P=0.5, R=1.0) -> F0.5 = 5/9 = 0.555555...
    score_f = compute_entity_f05({"S2-001"}, {"S2-001", "S3-002"})
    expected_f = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0)  # 5/9 = 0.555555...
    assert abs(score_f - expected_f) < 1e-6, f"Test F failed: expected {expected_f}, got {score_f}"
    print(f"  [PASS] Test F (P=0.5, R=1.0): {score_f:.4f} (expected {expected_f:.4f})")

    # Test G: True 2 elems, Pred 3 elems (2 correct, 1 wrong: P=2/3, R=1.0) -> F0.5 = 5/7 = 0.714285...
    score_g = compute_entity_f05({"S2-001", "S3-002"}, {"S2-001", "S2-002", "S3-002"})
    expected_g = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0)  # 5/7 = 0.714285...
    assert abs(score_g - expected_g) < 1e-6, f"Test G failed: expected {expected_g}, got {score_g}"
    print(f"  [PASS] Test G (P=2/3, R=1.0): {score_g:.4f} (expected {expected_g:.4f})")

    print("\nALL METRIC UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-dummy":
        run_unit_tests()
    else:
        print("Usage: python metrics.py --test-dummy")
