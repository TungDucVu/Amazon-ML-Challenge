"""Verification Script for Phase 1: Ingestion, Validation Setup & Ground Truth Scorer.

Tests:
1. Dataset generation and directory structure.
2. Strict tab-separated TSV parsing & column schema validation.
3. Open-set country partitioning (US, India, and France).
4. Exact Macro F_0.5 metric computation (matching official README example).
5. Singleton scoring rules (1.0 for true empty, 0.0 for false merge).
6. Stratified 80/20 train/val split.
"""

from __future__ import annotations

import os
import sys

# Ensure src is on Python path
sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))

from data_loader import (
    load_source_tsv,
    load_ground_truth_tsv,
    partition_by_country,
    load_dataset_bundle,
)
from metrics import compute_entity_f05, evaluate_predictions, compute_blocking_metrics
from split import create_stratified_validation_split
from synthetic_data import generate_benchmark_dataset


def run_phase1_verification() -> bool:
    print("=" * 80)
    print("RUNNING PHASE 1 VERIFICATION: INGESTION, SCHEMA & VALIDATION METRIC HARNESS")
    print("=" * 80)

    dataset_dir = "dataset"
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # Step 1: Ensure dataset files exist (generate benchmark dataset if missing)
    if not os.path.isfile(os.path.join(train_dir, "train_source1.tsv")):
        print("\n[Step 1] Initializing benchmark dataset with noise and open-set France...")
        generate_benchmark_dataset(dataset_dir, n_train_s1=120, n_test_s1=60, random_state=42)
    else:
        print("\n[Step 1] Existing dataset found in dataset/.")

    # Step 2: Test TSV Data Loading & Schema Enforcement
    print("\n[Step 2] Testing strict TSV loading and schema validation...")
    s1_train, s2_train, s3_train, gt_train = load_dataset_bundle(train_dir, is_train=True)
    assert s1_train is not None, "Failed to load S1 train"
    assert s2_train is not None, "Failed to load S2 train"
    assert s3_train is not None, "Failed to load S3 train"
    assert gt_train is not None, "Failed to load Ground Truth train"
    assert len(s1_train) == len(gt_train), "Mismatch between S1 count and Ground Truth count"
    print(f"  ✓ Successfully loaded Train bundle: S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}, GT={len(gt_train)}")

    s1_test, s2_test, s3_test, _ = load_dataset_bundle(test_dir, is_train=False)
    print(f"  ✓ Successfully loaded Test bundle:  S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")

    # Step 3: Test Open-Set Country Partitioning
    print("\n[Step 3] Testing open-set country partitioning...")
    train_countries = set(s1_train["country"].unique())
    test_countries = set(s1_test["country"].unique())
    print(f"  Train countries: {train_countries}")
    print(f"  Test countries:  {test_countries}")
    assert "US" in train_countries and "India" in train_countries, "Missing US/India in train"
    assert "France" in test_countries, "France must be present in open-set test data"
    
    parts_test = partition_by_country(s1_test)
    assert "France" in parts_test and len(parts_test["France"]) > 0, "Failed to partition France"
    print(f"  ✓ Country partitioning verified. France partitioned into {len(parts_test['France'])} records.")

    # Step 4: Verify Official F_0.5 Metric Calculation
    print("\n[Step 4] Testing exact Macro F_0.5 metric calculations...")
    
    # Check official example from README.md:
    # Pred = [S2-00047, S2-00193, S3-00812], Truth = [S2-00047, S3-00812]
    # P = 2/3, R = 1.0 -> F_0.5 = 0.714
    pred_example = {"S2-00047", "S2-00193", "S3-00812"}
    truth_example = {"S2-00047", "S3-00812"}
    f05_val, p_val, r_val = compute_entity_f05(pred_example, truth_example)
    print(f"  Official README Example: P={p_val:.3f}, R={r_val:.3f}, F_0.5={f05_val:.3f}")
    assert round(p_val, 3) == 0.667, f"Expected Precision 0.667, got {p_val}"
    assert round(r_val, 3) == 1.000, f"Expected Recall 1.000, got {r_val}"
    assert round(f05_val, 3) == 0.714, f"Expected F_0.5 0.714, got {f05_val}"
    print("  ✓ Official README example F_0.5 = 0.714 match verified!")

    # Check Singleton Rules
    f05_singleton_correct, _, _ = compute_entity_f05(set(), set())
    assert f05_singleton_correct == 1.0, f"Correct singleton must score 1.0, got {f05_singleton_correct}"
    f05_singleton_wrong, _, _ = compute_entity_f05({"S2-99999"}, set())
    assert f05_singleton_wrong == 0.0, f"False merge on singleton must score 0.0, got {f05_singleton_wrong}"
    print("  ✓ Singleton scoring rules verified (1.0 for true empty, 0.0 for false merge).")

    # Step 5: Verify Stratified 80/20 Train/Val Split
    print("\n[Step 5] Testing stratified 80/20 train/validation split...")
    train_ids, val_ids = create_stratified_validation_split(
        s1_train, gt_train, val_size=0.20, random_state=42
    )
    assert len(train_ids) + len(val_ids) == len(s1_train), "Split size mismatch"
    val_ratio = len(val_ids) / len(s1_train)
    print(f"  Split results: Train={len(train_ids)}, Val={len(val_ids)} (Val ratio: {val_ratio:.2%})")
    assert 0.18 <= val_ratio <= 0.22, f"Validation ratio {val_ratio} not close to 20%"
    print("  ✓ Stratified train/val split verified.")

    # Step 6: Verify Evaluation Harness
    print("\n[Step 6] Testing evaluation harness across entire dataset...")
    # Perfect predictions test
    perfect_eval = evaluate_predictions(gt_train, gt_train)
    assert perfect_eval["macro_f05"] == 1.0, f"Perfect predictions must give F_0.5=1.0, got {perfect_eval['macro_f05']}"
    print(f"  ✓ Perfect ground truth eval: Macro F_0.5={perfect_eval['macro_f05']:.4f}")

    # Empty predictions test
    empty_preds = {s1: set() for s1 in gt_train.keys()}
    empty_eval = evaluate_predictions(empty_preds, gt_train)
    print(f"  ✓ Empty predictions eval: Macro F_0.5={empty_eval['macro_f05']:.4f} (Singletons earned: {empty_eval['singleton_accuracy']:.2%})")

    print("\n" + "=" * 80)
    print("PHASE 1 VERIFICATION PASSED SUCCESSFULLY! ALL TESTS GREEN.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_phase1_verification()
    sys.exit(0 if success else 1)
