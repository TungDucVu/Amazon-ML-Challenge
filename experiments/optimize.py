import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))

from data_loader import load_dataset_bundle
from split import create_stratified_validation_split
from blocking import generate_candidates
from metrics import compute_blocking_metrics, evaluate_predictions
from features import compute_features_for_pairs
from matcher import prepare_training_labels, train_lgbm_model, train_xgb_model, ensemble_predict, optimize_threshold_f05, apply_threshold_to_predictions

def optimize():
    data_dir = "dataset"
    train_dir = os.path.join(data_dir, "train")
    
    print("\n--- PHASE D: VALIDATION DESIGN ---")
    s1_train, s2_train, s3_train, gt_train = load_dataset_bundle(train_dir, is_train=True)
    train_ids, val_ids = create_stratified_validation_split(s1_train, gt_train, val_size=0.20, random_state=42)
    
    print(f"Validation size: {len(val_ids)} S1 entities")
    
    val_s1 = s1_train[s1_train['entity_id'].isin(val_ids)].copy()
    print("Country distribution in Val:")
    print(val_s1['country'].value_counts().to_string())
    
    print("\n--- PHASE E: BLOCKING OPTIMIZATION ---")
    targets_train = pd.concat([s2_train, s3_train], ignore_index=True)
    
    # We evaluate blocking on the validation set.
    val_candidates = generate_candidates(
        val_s1, s2_train, s3_train,
        tfidf_top_k=15, similarity_threshold=0.10,
        max_candidates_per_s1=25, min_candidates_per_s1=5,
        show_progress=False
    )
    
    blocking_metrics = compute_blocking_metrics(val_candidates, gt_train)
    print(f"Candidate Recall: {blocking_metrics['blocking_recall_ceiling']:.4f}")
    print(f"Average Candidates/S1: {blocking_metrics['avg_candidates_per_s1']:.1f}")
    print(f"Max Candidates/S1: {blocking_metrics['max_candidates_per_s1']}")
    
    print("\n--- PHASE G & H: MODEL COMPARISON & THRESHOLD OPTIMIZATION ---")
    # Generate train candidates (excluding val ids for strict isolation)
    train_s1 = s1_train[s1_train['entity_id'].isin(train_ids)].copy()
    train_candidates = generate_candidates(
        train_s1, s2_train, s3_train,
        tfidf_top_k=15, similarity_threshold=0.10,
        max_candidates_per_s1=25, min_candidates_per_s1=5,
        show_progress=False
    )
    
    train_features_df, train_pair_ids = compute_features_for_pairs(train_s1, targets_train, train_candidates, show_progress=False)
    val_features_df, val_pair_ids = compute_features_for_pairs(val_s1, targets_train, val_candidates, show_progress=False)
    
    y_train = prepare_training_labels(train_pair_ids, gt_train)
    y_val = prepare_training_labels(val_pair_ids, gt_train)
    
    X_train = train_features_df.values
    X_val = val_features_df.values
    
    print("Training LightGBM...")
    lgbm = train_lgbm_model(X_train, y_train, X_val, y_val)
    print("Training XGBoost...")
    xgb = train_xgb_model(X_train, y_train, X_val, y_val)
    
    models = {
        "LightGBM": lgbm.predict(X_val),
        "XGBoost": xgb.predict(import_xgb().DMatrix(X_val)) if xgb else None,
        "Ensemble (0.5/0.5)": ensemble_predict(lgbm, xgb, X_val, lgbm_weight=0.5)
    }
    
    for name, probs in models.items():
        if probs is None: continue
        best_th, best_f05 = optimize_threshold_f05(
            probs, val_pair_ids, gt_train, val_ids,
            threshold_range=(0.2, 0.95), n_steps=60
        )
        preds = apply_threshold_to_predictions(probs, val_pair_ids, best_th, val_ids)
        eval_res = evaluate_predictions(preds, gt_train, required_s1_ids=val_ids)
        print(f"\nModel: {name}")
        print(f"  Best Threshold: {best_th:.4f}")
        print(f"  Macro F_0.5: {best_f05:.4f}")
        print(f"  Precision: {eval_res['macro_precision']:.4f}")
        print(f"  Recall: {eval_res['macro_recall']:.4f}")

def import_xgb():
    import xgboost as xgb
    return xgb

if __name__ == '__main__':
    optimize()
