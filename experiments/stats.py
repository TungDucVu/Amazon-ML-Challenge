import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
from data_loader import load_dataset_bundle
from split import create_stratified_validation_split
from blocking import generate_candidates
import numpy as np

train_dir = "dataset/train"
s1_train, s2_train, s3_train, gt_train = load_dataset_bundle(train_dir, is_train=True)

train_candidates = generate_candidates(
    s1_train, s2_train, s3_train,
    tfidf_top_k=15, similarity_threshold=0.10,
    max_candidates_per_s1=25, min_candidates_per_s1=5,
    show_progress=False
)

counts = [len(v) for v in train_candidates.values()]
print(f"Average: {np.mean(counts):.2f}")
print(f"Median: {np.median(counts):.2f}")
print(f"P95: {np.percentile(counts, 95):.2f}")
print(f"Max: {np.max(counts):.2f}")
