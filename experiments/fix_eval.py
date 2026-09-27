import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
from data_loader import load_dataset_bundle
from split import create_stratified_validation_split
from blocking import generate_candidates
from metrics import compute_blocking_metrics
import pandas as pd

train_dir = "dataset/train"
s1_train, s2_train, s3_train, gt_train = load_dataset_bundle(train_dir, is_train=True)
train_ids, val_ids = create_stratified_validation_split(s1_train, gt_train, val_size=0.20, random_state=42)

val_s1 = s1_train[s1_train['entity_id'].isin(val_ids)].copy()
val_candidates = generate_candidates(
    val_s1, s2_train, s3_train,
    tfidf_top_k=15, similarity_threshold=0.10,
    max_candidates_per_s1=25, min_candidates_per_s1=5,
    show_progress=False
)

val_gt = {k: v for k, v in gt_train.items() if k in val_ids}
blocking_metrics = compute_blocking_metrics(val_candidates, val_gt)
print(f"Validation Candidate Recall: {blocking_metrics['blocking_recall_ceiling']:.4f}")
print(f"Average Candidates/S1: {blocking_metrics['avg_candidates_per_s1']:.1f}")
print(f"Captured: {blocking_metrics['captured_true_matches']} / {blocking_metrics['total_true_matches']}")

