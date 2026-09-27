"""Machine Learning Matching Classifier and Threshold Optimization.

GBDT ensemble using LightGBM and XGBoost for binary match/non-match classification.
Optimizes decision threshold specifically for Macro F_0.5 (precision-heavy).

Key Design:
- Binary classification: true ground truth pairs = 1, non-matching candidates = 0.
- Class imbalance handled via scale_pos_weight.
- Threshold θ* calibrated on validation set to maximize Macro F_0.5.
- Conservative thresholds (0.65-0.80) to minimize false merges.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from metrics import compute_entity_f05, evaluate_predictions


def prepare_training_labels(
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
) -> np.ndarray:
    """Create binary labels for candidate pairs based on ground truth.

    Args:
        pair_ids: List of (s1_id, candidate_id) tuples.
        ground_truth: Dict mapping s1_id -> set of true matched IDs.

    Returns:
        Binary label array (1 = match, 0 = non-match).
    """
    labels = np.zeros(len(pair_ids), dtype=np.float32)
    for i, (s1_id, cand_id) in enumerate(pair_ids):
        true_matches = ground_truth.get(s1_id, set())
        if cand_id in true_matches:
            labels[i] = 1.0
    return labels


def train_lgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> lgb.Booster:
    """Train a LightGBM binary classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (0/1).
        X_val: Validation features (for early stopping).
        y_val: Validation labels.
        params: LightGBM hyperparameters (optional, defaults provided).

    Returns:
        Trained LightGBM Booster model.
    """
    # Compute class imbalance ratio
    n_pos = max(y_train.sum(), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    scale_pos_weight = n_neg / n_pos

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 10,
        "verbose": -1,
        "seed": 42,
    }

    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train)

    callbacks = [lgb.log_evaluation(period=0)]  # Suppress per-iteration logs
    valid_sets = [dtrain]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

    model = lgb.train(
        default_params,
        dtrain,
        num_boost_round=500,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    return model


def train_xgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> xgb.Booster:
    """Train an XGBoost binary classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (0/1).
        X_val: Validation features (for early stopping).
        y_val: Validation labels.
        params: XGBoost hyperparameters (optional, defaults provided).

    Returns:
        Trained XGBoost Booster model.
    """
    n_pos = max(y_train.sum(), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    scale_pos_weight = n_neg / n_pos

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "min_child_weight": 5,
        "seed": 42,
        "verbosity": 0,
    }

    if params:
        default_params.update(params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]

    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "val"))

    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30 if X_val is not None else None,
        verbose_eval=False,
    )

    return model


def ensemble_predict(
    lgbm_model: lgb.Booster,
    xgb_model: xgb.Booster,
    X: np.ndarray,
    lgbm_weight: float = 0.5,
) -> np.ndarray:
    """Generate ensemble predictions from LightGBM and XGBoost.

    Args:
        lgbm_model: Trained LightGBM model.
        xgb_model: Trained XGBoost model.
        X: Feature matrix.
        lgbm_weight: Weight for LightGBM predictions (XGBoost gets 1 - weight).

    Returns:
        Ensemble match probability array.
    """
    lgbm_preds = lgbm_model.predict(X)
    xgb_preds = xgb_model.predict(xgb.DMatrix(X))

    xgb_weight = 1.0 - lgbm_weight
    return lgbm_weight * lgbm_preds + xgb_weight * xgb_preds


def optimize_threshold_f05(
    probabilities: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    s1_ids: List[str],
    threshold_range: Tuple[float, float] = (0.3, 0.95),
    n_steps: int = 50,
) -> Tuple[float, float]:
    """Search for the optimal threshold maximizing Macro F_0.5 on validation set.

    Because F_0.5 weights precision 2x over recall, the optimal threshold is
    typically conservative (0.65-0.80) to avoid false merges.

    Args:
        probabilities: Match probability for each candidate pair.
        pair_ids: List of (s1_id, candidate_id) tuples.
        ground_truth: Ground truth dict.
        s1_ids: List of S1 entity IDs being evaluated.
        threshold_range: (min_threshold, max_threshold).
        n_steps: Number of threshold values to try.

    Returns:
        Tuple of (optimal_threshold, best_f05_score).
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    best_threshold = 0.5
    best_f05 = 0.0

    for theta in thresholds:
        # Build prediction dict from threshold
        predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_ids}
        for i, (s1_id, cand_id) in enumerate(pair_ids):
            if s1_id in predictions and probabilities[i] >= theta:
                predictions[s1_id].add(cand_id)

        # Compute macro F_0.5
        eval_result = evaluate_predictions(predictions, ground_truth, required_s1_ids=s1_ids)
        f05 = eval_result["macro_f05"]

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = theta

    return best_threshold, best_f05


def apply_threshold_to_predictions(
    probabilities: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    threshold: float,
    s1_entity_ids: List[str],
) -> Dict[str, Set[str]]:
    """Apply a decision threshold to generate final match predictions.

    Applies singleton safeguard: if no candidate exceeds threshold, output
    empty match list to earn 1.0 on true singletons.

    Args:
        probabilities: Match probability for each candidate pair.
        pair_ids: List of (s1_id, candidate_id) tuples.
        threshold: Decision threshold θ*.
        s1_entity_ids: All S1 entity IDs (ensures every S1 has a row).

    Returns:
        Dict mapping s1_id -> set of matched entity IDs.
    """
    predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_entity_ids}

    for i, (s1_id, cand_id) in enumerate(pair_ids):
        if s1_id in predictions and probabilities[i] >= threshold:
            predictions[s1_id].add(cand_id)

    return predictions
