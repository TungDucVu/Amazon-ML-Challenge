import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))

from data_loader import load_dataset_bundle
from split import create_stratified_validation_split
from blocking import generate_candidates
from features import compute_features_for_pairs
from matcher import prepare_training_labels, train_lgbm_model, train_xgb_model, ensemble_predict, optimize_threshold_f05, apply_threshold_to_predictions
from metrics import evaluate_predictions
from temp_features import compute_targeted_features

def run_ablation():
    print("\n--- ISSUE 6: MODEL ABLATION & ISSUE 5: TARGETED FEATURES ---")
    data_dir = "dataset"
    train_dir = os.path.join(data_dir, "train")
    
    s1_train, s2_train, s3_train, gt_train = load_dataset_bundle(train_dir, is_train=True)
    targets_train = pd.concat([s2_train, s3_train], ignore_index=True)
    
    train_ids, val_ids = create_stratified_validation_split(s1_train, gt_train, val_size=0.20, random_state=42)
    val_s1 = s1_train[s1_train['entity_id'].isin(val_ids)].copy()
    train_s1 = s1_train[s1_train['entity_id'].isin(train_ids)].copy()
    
    train_cands = generate_candidates(
        train_s1, s2_train, s3_train, tfidf_top_k=15, similarity_threshold=0.10, max_candidates_per_s1=25, min_candidates_per_s1=5, show_progress=False
    )
    val_cands = generate_candidates(
        val_s1, s2_train, s3_train, tfidf_top_k=15, similarity_threshold=0.10, max_candidates_per_s1=25, min_candidates_per_s1=5, show_progress=False
    )
    
    train_feat, train_pids = compute_features_for_pairs(train_s1, targets_train, train_cands, show_progress=False)
    val_feat, val_pids = compute_features_for_pairs(val_s1, targets_train, val_cands, show_progress=False)
    
    y_train = prepare_training_labels(train_pids, gt_train)
    y_val = prepare_training_labels(val_pids, gt_train)
    
    # Compute targeted features
    def augment_features(df_s1, cands, pids, orig_feats):
        s1_lookup = df_s1.set_index('entity_id')
        tgt_lookup = targets_train.set_index('entity_id')
        
        new_rows = []
        for s1, cand in pids:
            r1 = s1_lookup.loc[s1]
            r2 = tgt_lookup.loc[cand]
            new_feats = compute_targeted_features(
                r1['business_name'], r1['business_address'], r1['country'],
                r2['business_name'], r2['business_address'], r2['country']
            )
            new_rows.append(new_feats)
        
        new_df = pd.DataFrame(new_rows)
        return pd.concat([orig_feats, new_df], axis=1)

    train_feat_aug = augment_features(train_s1, train_cands, train_pids, train_feat)
    val_feat_aug = augment_features(val_s1, val_cands, val_pids, val_feat)
    
    models_to_run = [
        ("A. Current LightGBM", train_feat, val_feat, "lgb"),
        ("B. Current XGBoost", train_feat, val_feat, "xgb"),
        ("C. Current 50/50 ensemble", train_feat, val_feat, "ens"),
        ("D. Ensemble + Targeted Features", train_feat_aug, val_feat_aug, "ens"),
    ]
    
    results = []
    
    for name, t_X, v_X, m_type in models_to_run:
        lgbm = train_lgbm_model(t_X.values, y_train, v_X.values, y_val)
        xgb_mod = train_xgb_model(t_X.values, y_train, v_X.values, y_val)
        
        if m_type == "lgb":
            probs = lgbm.predict(v_X.values)
        elif m_type == "xgb":
            import xgboost as xgb
            probs = xgb_mod.predict(xgb.DMatrix(v_X.values))
        else:
            probs = ensemble_predict(lgbm, xgb_mod, v_X.values, lgbm_weight=0.5)
            
        best_th, best_f05 = optimize_threshold_f05(probs, val_pids, gt_train, val_ids, threshold_range=(0.2, 0.95), n_steps=60)
        preds = apply_threshold_to_predictions(probs, val_pids, best_th, val_ids)
        ev = evaluate_predictions(preds, gt_train, required_s1_ids=val_ids)
        
        fp = sum(len(preds.get(s, set()) - gt_train.get(s, set())) for s in val_ids)
        fn = sum(len(gt_train.get(s, set()) - preds.get(s, set())) for s in val_ids)
        
        results.append({
            'Model': name,
            'Feats': t_X.shape[1],
            'Th': best_th,
            'F05': best_f05,
            'Prec': ev['macro_precision'],
            'Rec': ev['macro_recall'],
            'FP': fp,
            'FN': fn
        })
        
    print("\nModel | Features | Threshold | Macro F0.5 | Precision | Recall | FP | FN")
    for r in results:
        print(f"{r['Model']} | {r['Feats']} | {r['Th']:.4f} | {r['F05']:.4f} | {r['Prec']:.4f} | {r['Rec']:.4f} | {r['FP']} | {r['FN']}")

if __name__ == '__main__':
    run_ablation()
