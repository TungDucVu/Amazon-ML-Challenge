"""End-to-End Business Entity Resolution Pipeline.

Main entry point that orchestrates all phases:
  Phase 1: Data loading, validation split, metric harness
  Phase 2: Candidate generation (blocking)
  Phase 3: Feature engineering
  Phase 4: ML model training & threshold optimization
  Phase 5: Post-processing & output generation

Usage:
    python run_pipeline.py --data-dir dataset --output-dir output [--mode train|predict|both]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Set

import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))

from data_loader import load_dataset_bundle, partition_by_country
from split import create_stratified_validation_split
from metrics import evaluate_predictions, compute_blocking_metrics
from blocking import generate_candidates, write_candidate_pairs_tsv
from features import compute_features_for_pairs
from matcher import (
    prepare_training_labels,
    train_lgbm_model,
    train_xgb_model,
    ensemble_predict,
    optimize_threshold_f05,
    apply_threshold_to_predictions,
)
from postprocessing import (
    write_matching_results_tsv,
    verify_subset_constraint,
    verify_output_format,
    run_official_validator,
)


def run_full_pipeline(
    data_dir: str = "dataset",
    output_dir: str = "output",
    generate_sample: bool = False,
) -> Dict:
    """Run the full entity resolution pipeline.

    Args:
        data_dir: Root dataset directory (containing train/ and test/).
        output_dir: Output directory for TSV files.
        generate_sample: If True, generate synthetic sample data first.

    Returns:
        Dict of evaluation results and diagnostics.
    """
    results = {}
    t_start = time.time()

    # ==========================================
    # PHASE 1: Data Loading & Validation Setup
    # ==========================================
    print("=" * 80)
    print("PHASE 1: DATA LOADING & VALIDATION SETUP")
    print("=" * 80)

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if generate_sample or not os.path.isfile(os.path.join(train_dir, "train_source1.tsv")):
        print("[Phase 1] Generating sample benchmark dataset...")
        from synthetic_data import generate_benchmark_dataset
        generate_benchmark_dataset(data_dir, n_train_s1=120, n_test_s1=60)

    print("[Phase 1] Loading training data...")
    s1_train, s2_train, s3_train, gt_train = load_dataset_bundle(train_dir, is_train=True)
    print(f"  Train: S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}, GT={len(gt_train)}")

    print("[Phase 1] Loading test data...")
    s1_test, s2_test, s3_test, _ = load_dataset_bundle(test_dir, is_train=False)
    print(f"  Test:  S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")

    # Stratified Train/Val Split
    print("[Phase 1] Creating stratified 80/20 train/val split...")
    train_s1_ids, val_s1_ids = create_stratified_validation_split(
        s1_train, gt_train, val_size=0.20, random_state=42
    )
    print(f"  Split: Train={len(train_s1_ids)}, Val={len(val_s1_ids)}")

    # Country summary
    train_countries = sorted(s1_train["country"].unique())
    test_countries = sorted(s1_test["country"].unique())
    print(f"  Train countries: {train_countries}")
    print(f"  Test countries:  {test_countries}")

    # ==========================================
    # PHASE 2: CANDIDATE GENERATION (BLOCKING)
    # ==========================================
    print("\n" + "=" * 80)
    print("PHASE 2: CANDIDATE GENERATION (BLOCKING)")
    print("=" * 80)

    print("[Phase 2] Generating train candidates (for model training)...")
    train_candidates = generate_candidates(
        s1_train, s2_train, s3_train,
        tfidf_top_k=10,
        similarity_threshold=0.12,
        max_candidates_per_s1=25,
        min_candidates_per_s1=5,
    )

    # Evaluate blocking quality on training set
    blocking_metrics = compute_blocking_metrics(train_candidates, gt_train)
    print(f"  [Blocking Quality]")
    print(f"    Recall Ceiling: {blocking_metrics['blocking_recall_ceiling']:.4f}")
    print(f"    Captured Matches: {blocking_metrics['captured_true_matches']}/{blocking_metrics['total_true_matches']}")
    print(f"    Avg Candidates/S1: {blocking_metrics['avg_candidates_per_s1']:.1f}")
    print(f"    Max Candidates/S1: {blocking_metrics['max_candidates_per_s1']}")
    results["blocking"] = blocking_metrics

    print("\n[Phase 2] Generating test candidates...")
    test_candidates = generate_candidates(
        s1_test, s2_test, s3_test,
        tfidf_top_k=10,
        similarity_threshold=0.12,
        max_candidates_per_s1=25,
        min_candidates_per_s1=5,
    )

    # Write candidate_pairs.tsv for test set
    candidate_output = os.path.join(output_dir, "candidate_pairs.tsv")
    test_s1_ids = s1_test["entity_id"].tolist()
    write_candidate_pairs_tsv(test_candidates, candidate_output, test_s1_ids)
    print(f"  ✓ Written: {candidate_output}")

    # ==========================================
    # PHASE 3: FEATURE ENGINEERING
    # ==========================================
    print("\n" + "=" * 80)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 80)

    # Combine S2+S3 for target lookups
    targets_train = pd.concat([s2_train, s3_train], ignore_index=True)
    targets_test = pd.concat([s2_test, s3_test], ignore_index=True)

    print("[Phase 3] Computing features for training candidate pairs...")
    train_features_df, train_pair_ids = compute_features_for_pairs(
        s1_train, targets_train, train_candidates, show_progress=True
    )
    print(f"  Train features: {train_features_df.shape[0]} pairs × {train_features_df.shape[1]} features")

    print("[Phase 3] Computing features for test candidate pairs...")
    test_features_df, test_pair_ids = compute_features_for_pairs(
        s1_test, targets_test, test_candidates, show_progress=True
    )
    print(f"  Test features:  {test_features_df.shape[0]} pairs × {test_features_df.shape[1]} features")

    feature_names = list(train_features_df.columns)
    print(f"  Feature names: {feature_names}")

    # ==========================================
    # PHASE 4: ML MODEL TRAINING & THRESHOLD
    # ==========================================
    print("\n" + "=" * 80)
    print("PHASE 4: ML MODEL TRAINING & THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Prepare labels
    train_labels = prepare_training_labels(train_pair_ids, gt_train)
    print(f"  Labels: {int(train_labels.sum())} positive, {int(len(train_labels) - train_labels.sum())} negative")

    # Split features into train/val based on S1 ID membership
    val_s1_set = set(val_s1_ids)
    train_mask = np.array([pid[0] not in val_s1_set for pid in train_pair_ids])
    val_mask = ~train_mask

    X_train_split = train_features_df.values[train_mask]
    y_train_split = train_labels[train_mask]
    X_val_split = train_features_df.values[val_mask]
    y_val_split = train_labels[val_mask]

    val_pair_ids = [pid for pid, m in zip(train_pair_ids, val_mask) if m]

    print(f"  Train split: {len(X_train_split)} pairs ({int(y_train_split.sum())} pos)")
    print(f"  Val split:   {len(X_val_split)} pairs ({int(y_val_split.sum())} pos)")

    # Train LightGBM
    print("\n[Phase 4] Training LightGBM model...")
    lgbm_model = train_lgbm_model(X_train_split, y_train_split, X_val_split, y_val_split)
    print(f"  ✓ LightGBM trained ({lgbm_model.num_trees()} trees)")

    # Train XGBoost
    print("[Phase 4] Training XGBoost model...")
    xgb_model = train_xgb_model(X_train_split, y_train_split, X_val_split, y_val_split)
    print(f"  ✓ XGBoost trained ({xgb_model.best_iteration} best iteration)")

    # Ensemble predictions on validation set
    print("[Phase 4] Computing ensemble predictions on validation set...")
    val_probs = ensemble_predict(lgbm_model, xgb_model, X_val_split, lgbm_weight=0.5)

    # Optimize threshold for Macro F_0.5
    print("[Phase 4] Optimizing threshold for Macro F_0.5 on validation set...")
    best_threshold, best_val_f05 = optimize_threshold_f05(
        val_probs, val_pair_ids, gt_train, val_s1_ids,
        threshold_range=(0.2, 0.95), n_steps=60,
    )
    print(f"  ✓ Optimal Threshold: θ* = {best_threshold:.4f}")
    print(f"  ✓ Validation Macro F_0.5: {best_val_f05:.4f}")
    results["threshold"] = best_threshold
    results["val_f05"] = best_val_f05

    # Full validation evaluation at optimal threshold
    val_predictions = apply_threshold_to_predictions(
        val_probs, val_pair_ids, best_threshold, val_s1_ids
    )
    val_eval = evaluate_predictions(val_predictions, gt_train, required_s1_ids=val_s1_ids)
    print(f"\n  [Validation Results at θ*={best_threshold:.3f}]")
    print(f"    Macro F_0.5:        {val_eval['macro_f05']:.4f}")
    print(f"    Macro Precision:    {val_eval['macro_precision']:.4f}")
    print(f"    Macro Recall:       {val_eval['macro_recall']:.4f}")
    print(f"    Singleton Accuracy: {val_eval['singleton_accuracy']:.4f}")
    print(f"    Non-Singleton F_0.5:{val_eval['non_singleton_f05']:.4f}")
    results["val_eval"] = val_eval

    # ==========================================
    # PHASE 5: POST-PROCESSING & OUTPUT
    # ==========================================
    print("\n" + "=" * 80)
    print("PHASE 5: POST-PROCESSING & OUTPUT GENERATION")
    print("=" * 80)

    # Retrain models on FULL training data for test predictions
    print("[Phase 5] Retraining models on full training data...")
    X_full = train_features_df.values
    y_full = train_labels

    lgbm_full = train_lgbm_model(X_full, y_full)
    xgb_full = train_xgb_model(X_full, y_full)
    print(f"  ✓ LightGBM retrained ({lgbm_full.num_trees()} trees)")
    print(f"  ✓ XGBoost retrained")

    # Predict on test set
    print("[Phase 5] Generating test predictions...")
    if len(test_features_df) > 0:
        test_probs = ensemble_predict(lgbm_full, xgb_full, test_features_df.values, lgbm_weight=0.5)
    else:
        test_probs = np.array([])

    test_predictions = apply_threshold_to_predictions(
        test_probs, test_pair_ids, best_threshold, test_s1_ids
    )

    # Verify output format
    print("[Phase 5] Verifying output format...")
    format_errors = verify_output_format(test_predictions, test_s1_ids)
    if format_errors:
        print(f"  ⚠ Format errors: {format_errors}")
    else:
        print("  ✓ Output format verified (no errors)")

    # Verify subset constraint
    subset_violations = verify_subset_constraint(test_predictions, test_candidates)
    if subset_violations:
        print(f"  ⚠ Subset violations: {subset_violations[:3]}")
    else:
        print("  ✓ Subset constraint verified (matching ⊆ candidate)")

    # Write matching_results.tsv
    matching_output = os.path.join(output_dir, "matching_results.tsv")
    write_matching_results_tsv(test_predictions, matching_output, test_s1_ids)
    print(f"  ✓ Written: {matching_output}")

    # Output statistics
    n_singletons = sum(1 for s1 in test_s1_ids if len(test_predictions.get(s1, set())) == 0)
    n_matched = len(test_s1_ids) - n_singletons
    avg_matches = (
        sum(len(test_predictions.get(s1, set())) for s1 in test_s1_ids) / len(test_s1_ids)
        if test_s1_ids else 0
    )
    print(f"\n  [Test Output Statistics]")
    print(f"    Total S1 entities: {len(test_s1_ids)}")
    print(f"    Predicted singletons: {n_singletons}")
    print(f"    Predicted matched: {n_matched}")
    print(f"    Avg matches per S1: {avg_matches:.2f}")

    # Run official validator
    print("\n[Phase 5] Running official validate_submission.py...")
    validator_path = os.path.join(
        "6ab10eb3b23ba_student_resource", "student_resource", "utils", "validate_submission.py"
    )
    if os.path.isfile(validator_path):
        returncode, validator_output = run_official_validator(
            matching_output, candidate_output, test_dir, validator_path
        )
        print(f"  {validator_output.strip()}")
        if returncode == 0:
            print("  ✓ OFFICIAL VALIDATOR: PASS")
        else:
            print("  ✗ OFFICIAL VALIDATOR: FAIL")
        results["validator_pass"] = (returncode == 0)
    else:
        print(f"  ⚠ Validator not found at {validator_path}")
        results["validator_pass"] = None

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"  Total Runtime:        {t_elapsed:.1f}s")
    print(f"  Blocking Recall:      {blocking_metrics['blocking_recall_ceiling']:.4f}")
    print(f"  Avg Candidates/S1:    {blocking_metrics['avg_candidates_per_s1']:.1f}")
    print(f"  Optimal Threshold:    {best_threshold:.4f}")
    print(f"  Val Macro F_0.5:      {best_val_f05:.4f}")
    print(f"  Output files:")
    print(f"    {matching_output}")
    print(f"    {candidate_output}")
    results["runtime_seconds"] = t_elapsed

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default="dataset", help="Dataset root directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate synthetic sample dataset (for testing pipeline)",
    )
    args = parser.parse_args()

    results = run_full_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        generate_sample=args.generate_sample,
    )
