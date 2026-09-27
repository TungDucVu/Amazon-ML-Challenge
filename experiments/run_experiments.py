import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))

from data_loader import load_dataset_bundle
from split import create_stratified_validation_split
from blocking import generate_candidates
from features import compute_features_for_pairs
from matcher import prepare_training_labels, train_lgbm_model, train_xgb_model, ensemble_predict, optimize_threshold_f05, apply_threshold_to_predictions
from metrics import compute_blocking_metrics, evaluate_predictions

import lightgbm as lgb
import xgboost as xgb

def run_k_sensitivity(s1_train, targets_train, s2_train, s3_train, gt_train, train_ids, val_ids):
    print("\n--- ISSUE 2: K SENSITIVITY EXPERIMENT ---")
    
    val_s1 = s1_train[s1_train['entity_id'].isin(val_ids)].copy()
    train_s1 = s1_train[s1_train['entity_id'].isin(train_ids)].copy()
    
    k_values = [5, 10, 15, 20, 25]
    
    results = []
    
    for k in k_values:
        print(f"\nEvaluating K={k}...")
        # 1. Generate Candidates
        val_cands = generate_candidates(
            val_s1, s2_train, s3_train,
            tfidf_top_k=k, similarity_threshold=0.10,
            max_candidates_per_s1=k, min_candidates_per_s1=5,
            show_progress=False
        )
        
        train_cands = generate_candidates(
            train_s1, s2_train, s3_train,
            tfidf_top_k=k, similarity_threshold=0.10,
            max_candidates_per_s1=k, min_candidates_per_s1=5,
            show_progress=False
        )
        
        val_gt = {eid: gt_train.get(eid, set()) for eid in val_ids}
        bm = compute_blocking_metrics(val_cands, val_gt)
        
        # 2. Features
        train_feat, train_pids = compute_features_for_pairs(train_s1, targets_train, train_cands, show_progress=False)
        val_feat, val_pids = compute_features_for_pairs(val_s1, targets_train, val_cands, show_progress=False)
        
        y_train = prepare_training_labels(train_pids, gt_train)
        y_val = prepare_training_labels(val_pids, gt_train)
        
        # 3. Model
        lgb_mod = train_lgbm_model(train_feat.values, y_train, val_feat.values, y_val)
        xgb_mod = train_xgb_model(train_feat.values, y_train, val_feat.values, y_val)
        probs = ensemble_predict(lgb_mod, xgb_mod, val_feat.values, lgbm_weight=0.5)
        
        # 4. Evaluate
        best_th, best_f05 = optimize_threshold_f05(probs, val_pids, gt_train, val_ids, threshold_range=(0.2, 0.95), n_steps=60)
        preds = apply_threshold_to_predictions(probs, val_pids, best_th, val_ids)
        ev = evaluate_predictions(preds, gt_train, required_s1_ids=val_ids)
        
        # False merges on singletons
        fp = 0
        fn = 0
        singleton_fp = 0
        for s1 in val_ids:
            p_set = preds.get(s1, set())
            t_set = gt_train.get(s1, set())
            fp += len(p_set - t_set)
            fn += len(t_set - p_set)
            if len(t_set) == 0 and len(p_set) > 0:
                singleton_fp += 1
                
        results.append({
            'K': k,
            'Recall': bm['blocking_recall_ceiling'],
            'Avg_Cand': bm['avg_candidates_per_s1'],
            'P95_Cand': np.percentile([len(c) for c in val_cands.values()], 95),
            'F05': best_f05,
            'Prec': ev['macro_precision'],
            'Rec': ev['macro_recall'],
            'FP': fp,
            'FN': fn,
            'Singleton_FP': singleton_fp
        })
    
    print("\nK | Cand Recall | Avg Cands | P95 | Macro F0.5 | Precision | Recall | FP | FN | Singleton FP")
    for r in results:
        print(f"{r['K']} | {r['Recall']:.4f} | {r['Avg_Cand']:.1f} | {r['P95_Cand']:.1f} | {r['F05']:.4f} | {r['Prec']:.4f} | {r['Rec']:.4f} | {r['FP']} | {r['FN']} | {r['Singleton_FP']}")
        
    return results

def run_threshold_sweep(val_feat, y_val, val_pids, gt_train, val_ids, lgbm, xgb):
    print("\n--- ISSUE 3: THRESHOLD EXPERIMENT ---")
    probs = ensemble_predict(lgbm, xgb, val_feat, lgbm_weight=0.5)
    
    thresholds = list(np.arange(0.05, 0.96, 0.025))
    res = []
    
    for th in thresholds:
        preds = apply_threshold_to_predictions(probs, val_pids, th, val_ids)
        ev = evaluate_predictions(preds, gt_train, required_s1_ids=val_ids)
        
        fp = 0
        fn = 0
        singleton_fp = 0
        for s1 in val_ids:
            p_set = preds.get(s1, set())
            t_set = gt_train.get(s1, set())
            fp += len(p_set - t_set)
            fn += len(t_set - p_set)
            if len(t_set) == 0 and len(p_set) > 0:
                singleton_fp += 1
                
        res.append((th, ev['macro_f05'], ev['macro_precision'], ev['macro_recall'], fp, fn, singleton_fp))
        
    res.sort(key=lambda x: -x[1])
    print("Top 10 Thresholds:")
    print("Threshold | Macro F0.5 | Precision | Recall | FP | FN | Singleton FP")
    for r in res[:10]:
        print(f"{r[0]:.3f} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]} | {r[5]} | {r[6]}")

def analyze_false_positives(val_feat_df, val_pids, probs, preds, gt_train, s1_train, targets_train):
    print("\n--- ISSUE 4: PRECISION-FOCUSED FEATURE ANALYSIS ---")
    
    s1_lookup = s1_train.set_index('entity_id')
    tgt_lookup = targets_train.set_index('entity_id')
    
    for idx, (s1, cand) in enumerate(val_pids):
        p_set = preds.get(s1, set())
        t_set = gt_train.get(s1, set())
        if cand in p_set and cand not in t_set:
            print("\nFALSE POSITIVE DETECTED:")
            r1 = s1_lookup.loc[s1]
            r2 = tgt_lookup.loc[cand]
            print(f"S1: {s1} | Name: {r1['business_name']} | Addr: {r1['business_address']} | Country: {r1['country']}")
            print(f"Cand: {cand} | Name: {r2['business_name']} | Addr: {r2['business_address']} | Country: {r2['country']}")
            
            feat_row = val_feat_df.iloc[idx]
            print(f"Important features: NameLev={feat_row['name_levenshtein_sim']:.2f}, AddrLev={feat_row['addr_levenshtein_sim']:.2f}, "
                  f"NumJaccard={feat_row['numeric_jaccard']:.2f}, Prob={probs[idx]:.3f}")

def run_validation_robustness(s1_train, targets_train, s2_train, s3_train, gt_train):
    print("\n--- ISSUE 7: VALIDATION ROBUSTNESS ---")
    seeds = [42, 52, 62, 72, 82]
    f05_scores = []
    
    for s in seeds:
        train_ids, val_ids = create_stratified_validation_split(s1_train, gt_train, val_size=0.20, random_state=s)
        val_s1 = s1_train[s1_train['entity_id'].isin(val_ids)].copy()
        train_s1 = s1_train[s1_train['entity_id'].isin(train_ids)].copy()
        
        val_cands = generate_candidates(
            val_s1, s2_train, s3_train,
            tfidf_top_k=15, similarity_threshold=0.10,
            max_candidates_per_s1=25, min_candidates_per_s1=5,
            show_progress=False
        )
        train_cands = generate_candidates(
            train_s1, s2_train, s3_train,
            tfidf_top_k=15, similarity_threshold=0.10,
            max_candidates_per_s1=25, min_candidates_per_s1=5,
            show_progress=False
        )
        
        train_feat, train_pids = compute_features_for_pairs(train_s1, targets_train, train_cands, show_progress=False)
        val_feat, val_pids = compute_features_for_pairs(val_s1, targets_train, val_cands, show_progress=False)
        
        y_train = prepare_training_labels(train_pids, gt_train)
        y_val = prepare_training_labels(val_pids, gt_train)
        
        lgb_mod = train_lgbm_model(train_feat.values, y_train, val_feat.values, y_val)
        xgb_mod = train_xgb_model(train_feat.values, y_train, val_feat.values, y_val)
        probs = ensemble_predict(lgb_mod, xgb_mod, val_feat.values, lgbm_weight=0.5)
        
        best_th, best_f05 = optimize_threshold_f05(probs, val_pids, gt_train, val_ids, threshold_range=(0.2, 0.95), n_steps=60)
        f05_scores.append(best_f05)
        print(f"Seed {s}: Macro F0.5 = {best_f05:.4f}")
        
    print(f"\nRobustness Summary: Mean F0.5 = {np.mean(f05_scores):.4f}, Std = {np.std(f05_scores):.4f}")

def main():
    train_dir = "dataset/train"
    s1_train, s2_train, s3_train, gt_train = load_dataset_bundle(train_dir, is_train=True)
    targets_train = pd.concat([s2_train, s3_train], ignore_index=True)
    
    # 1. K Sensitivity
    train_ids, val_ids = create_stratified_validation_split(s1_train, gt_train, val_size=0.20, random_state=42)
    run_k_sensitivity(s1_train, targets_train, s2_train, s3_train, gt_train, train_ids, val_ids)
    
    # 2. Get fixed K=25 predictions for Threshold & Error Analysis
    val_s1 = s1_train[s1_train['entity_id'].isin(val_ids)].copy()
    train_s1 = s1_train[s1_train['entity_id'].isin(train_ids)].copy()
    val_cands = generate_candidates(
        val_s1, s2_train, s3_train, tfidf_top_k=15, similarity_threshold=0.10, max_candidates_per_s1=25, min_candidates_per_s1=5, show_progress=False
    )
    train_cands = generate_candidates(
        train_s1, s2_train, s3_train, tfidf_top_k=15, similarity_threshold=0.10, max_candidates_per_s1=25, min_candidates_per_s1=5, show_progress=False
    )
    
    train_feat, train_pids = compute_features_for_pairs(train_s1, targets_train, train_cands, show_progress=False)
    val_feat, val_pids = compute_features_for_pairs(val_s1, targets_train, val_cands, show_progress=False)
    y_train = prepare_training_labels(train_pids, gt_train)
    y_val = prepare_training_labels(val_pids, gt_train)
    
    lgbm = train_lgbm_model(train_feat.values, y_train, val_feat.values, y_val)
    xgb = train_xgb_model(train_feat.values, y_train, val_feat.values, y_val)
    probs = ensemble_predict(lgbm, xgb, val_feat.values, lgbm_weight=0.5)
    
    # Run Threshold Sweep
    run_threshold_sweep(val_feat.values, y_val, val_pids, gt_train, val_ids, lgbm, xgb)
    
    # Run Error Analysis
    best_th, best_f05 = optimize_threshold_f05(probs, val_pids, gt_train, val_ids, threshold_range=(0.2, 0.95), n_steps=60)
    preds = apply_threshold_to_predictions(probs, val_pids, best_th, val_ids)
    analyze_false_positives(val_feat, val_pids, probs, preds, gt_train, s1_train, targets_train)
    
    # Validation Robustness
    run_validation_robustness(s1_train, targets_train, s2_train, s3_train, gt_train)

if __name__ == '__main__':
    main()
