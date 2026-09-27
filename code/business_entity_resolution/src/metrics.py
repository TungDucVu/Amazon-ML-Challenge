"""Official evaluation metrics for Amazon ML Challenge 2026.

Metric: Macro-averaged F_0.5 score across all Source 1 entities.
Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Weights Precision 2x higher than Recall to heavily penalize false merges.

Singleton Rule:
- True matches empty (singleton):
  - Predict empty -> 1.0
  - Predict any match -> 0.0
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple


def compute_entity_f05(
    predicted_ids: Set[str], true_ids: Set[str]
) -> Tuple[float, float, float]:
    """Compute F_0.5, Precision, and Recall for a single Source 1 entity.

    Args:
        predicted_ids: Set of predicted matching entity IDs (S2/S3).
        true_ids: Set of true matching entity IDs from ground truth.

    Returns:
        Tuple of (f05_score, precision, recall).
    """
    # Case 1: Singleton (true set is empty)
    if len(true_ids) == 0:
        if len(predicted_ids) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0

    # Case 2: Non-singleton, predicted empty
    if len(predicted_ids) == 0:
        return 0.0, 0.0, 0.0

    # Case 3: Non-singleton, predicted non-empty
    true_positives = len(predicted_ids & true_ids)
    if true_positives == 0:
        return 0.0, 0.0, 0.0

    precision = true_positives / len(predicted_ids)
    recall = true_positives / len(true_ids)

    denom = (0.25 * precision) + recall
    if denom == 0.0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return f05, precision, recall


def evaluate_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    required_s1_ids: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Evaluate predictions against ground truth across all Source 1 entities.

    Args:
        predictions: Dict mapping s1_id -> set of predicted matched IDs.
        ground_truth: Dict mapping s1_id -> set of ground truth matched IDs.
        required_s1_ids: Optional list of S1 IDs. If None, uses keys of ground_truth.

    Returns:
        Dict with keys:
            'macro_f05': Macro-averaged F_0.5 score
            'macro_precision': Macro-averaged Precision
            'macro_recall': Macro-averaged Recall
            'singleton_accuracy': Accuracy on 0-match entities
            'non_singleton_f05': Macro F_0.5 on entities with >= 1 true match
            'total_entities': Number of S1 entities evaluated
            'singleton_count': Number of true singletons
            'non_singleton_count': Number of true non-singletons
    """
    s1_ids = list(required_s1_ids) if required_s1_ids is not None else list(ground_truth.keys())

    f05_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    singleton_scores: List[float] = []
    non_singleton_f05: List[float] = []

    for s1_id in s1_ids:
        preds = predictions.get(s1_id, set())
        truth = ground_truth.get(s1_id, set())

        f05, prec, rec = compute_entity_f05(preds, truth)
        f05_scores.append(f05)
        precisions.append(prec)
        recalls.append(rec)

        if len(truth) == 0:
            singleton_scores.append(f05)
        else:
            non_singleton_f05.append(f05)

    n_total = len(s1_ids)
    if n_total == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "non_singleton_f05": 0.0,
            "total_entities": 0,
            "singleton_count": 0,
            "non_singleton_count": 0,
        }

    return {
        "macro_f05": sum(f05_scores) / n_total,
        "macro_precision": sum(precisions) / n_total,
        "macro_recall": sum(recalls) / n_total,
        "singleton_accuracy": (sum(singleton_scores) / len(singleton_scores)) if singleton_scores else 1.0,
        "non_singleton_f05": (sum(non_singleton_f05) / len(non_singleton_f05)) if non_singleton_f05 else 0.0,
        "total_entities": n_total,
        "singleton_count": len(singleton_scores),
        "non_singleton_count": len(non_singleton_f05),
    }


def compute_blocking_metrics(
    candidate_pairs: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Evaluate candidate generation (blocking) quality.

    Computes:
    - Recall ceiling: % of true matches captured in candidate sets
    - Average candidates per S1: audited efficiency budget (target 5-15)
    - Max candidates per S1
    - Reduction ratio: how much the total comparison space was cut

    Args:
        candidate_pairs: Dict mapping s1_id -> set of candidate IDs.
        ground_truth: Dict mapping s1_id -> set of true matched IDs.

    Returns:
        Dict with blocking diagnostics.
    """
    total_true_matches = 0
    captured_matches = 0
    candidate_counts: List[int] = []

    for s1_id, truth in ground_truth.items():
        candidates = candidate_pairs.get(s1_id, set())
        candidate_counts.append(len(candidates))

        total_true_matches += len(truth)
        captured_matches += len(truth & candidates)

    recall_ceiling = (captured_matches / total_true_matches) if total_true_matches > 0 else 1.0
    avg_candidates = (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0
    max_candidates = max(candidate_counts) if candidate_counts else 0

    return {
        "blocking_recall_ceiling": recall_ceiling,
        "captured_true_matches": captured_matches,
        "total_true_matches": total_true_matches,
        "avg_candidates_per_s1": avg_candidates,
        "max_candidates_per_s1": max_candidates,
        "total_s1_entities": len(ground_truth),
    }
