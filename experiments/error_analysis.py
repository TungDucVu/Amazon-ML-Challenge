import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
from data_loader import load_dataset_bundle
from split import create_stratified_validation_split
from blocking import generate_candidates
from features import compute_features_for_pairs
from matcher import prepare_training_labels, train_lgbm_model, ensemble_predict, apply_threshold_to_predictions
import pandas as pd

train_dir = "dataset/train"
s1_train, s2_train, s3_train, gt_train = load_dataset_bundle(train_dir, is_train=True)
train_ids, val_ids = create_stratified_validation_split(s1_train, gt_train, val_size=0.20, random_state=42)

targets_train = pd.concat([s2_train, s3_train], ignore_index=True)

train_s1 = s1_train[s1_train['entity_id'].isin(train_ids)].copy()
val_s1 = s1_train[s1_train['entity_id'].isin(val_ids)].copy()

train_candidates = generate_candidates(
    train_s1, s2_train, s3_train,
    tfidf_top_k=15, similarity_threshold=0.10,
    max_candidates_per_s1=25, min_candidates_per_s1=5,
    show_progress=False
)
val_candidates = generate_candidates(
    val_s1, s2_train, s3_train,
    tfidf_top_k=15, similarity_threshold=0.10,
    max_candidates_per_s1=25, min_candidates_per_s1=5,
    show_progress=False
)

train_features_df, train_pair_ids = compute_features_for_pairs(train_s1, targets_train, train_candidates, show_progress=False)
val_features_df, val_pair_ids = compute_features_for_pairs(val_s1, targets_train, val_candidates, show_progress=False)

y_train = prepare_training_labels(train_pair_ids, gt_train)
y_val = prepare_training_labels(val_pair_ids, gt_train)

lgbm = train_lgbm_model(train_features_df.values, y_train, val_features_df.values, y_val)
probs = lgbm.predict(val_features_df.values)

threshold = 0.3398
preds = apply_threshold_to_predictions(probs, val_pair_ids, threshold, val_ids)

fp = 0
fn = 0
print("--- ERROR ANALYSIS ---")
for s1, p_set in preds.items():
    t_set = gt_train.get(s1, set())
    fps = p_set - t_set
    fns = t_set - p_set
    if fps:
        print(f"FP: S1={s1} -> Predicted: {fps}")
        fp += len(fps)
    if fns:
        print(f"FN: S1={s1} -> Missed: {fns}")
        fn += len(fns)
print(f"\nTotal False Positives: {fp}")
print(f"Total False Negatives: {fn}")
