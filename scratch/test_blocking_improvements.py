import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import time

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from data_loader import load_source_file, load_ground_truth

def evaluate_recall(candidates_map, gt_map):
    total_true = 0
    retrieved_true = 0
    for s1_id, cands in candidates_map.items():
        if s1_id not in gt_map:
            continue
        true_set = set(gt_map[s1_id])
        num_true = len(true_set)
        total_true += num_true
        if num_true > 0:
            cand_set = set(cands)
            retrieved_true += len(true_set.intersection(cand_set))
    recall = (retrieved_true / total_true * 100) if total_true > 0 else 0
    avg_cands = np.mean([len(cands) for cands in candidates_map.values()])
    print(f"--> Target-Level Recall: {recall:.2f}% | Avg Candidates/S1: {avg_cands:.2f}", flush=True)
    return recall, avg_cands

if __name__ == "__main__":
    val_s1_path = os.path.abspath("dataset/split/val_split/val_source1.tsv")
    val_gt_path = os.path.abspath("dataset/split/val_split/val_ground_truth.tsv")
    s2_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv")
    s3_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv")

    print("Loading datasets...", flush=True)
    df_s1 = load_source_file(val_s1_path)
    df_gt = load_ground_truth(val_gt_path)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)

    # Sample 5,000 S1 queries for fast iteration
    np.random.seed(42)
    sample_s1_ids = set(np.random.choice(df_s1["entity_id"].values, size=5000, replace=False))
    df_s1_sub = df_s1[df_s1["entity_id"].isin(sample_s1_ids)].copy()

    gt_map = {
        row["source1_entity_id"]: [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        if row["matched_entity_ids"] else []
        for _, row in df_gt.iterrows()
        if row["source1_entity_id"] in sample_s1_ids
    }

    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    # Filter to US partition for rapid test
    df_s1_us = df_s1_sub[df_s1_sub["country"] == "US"].copy()
    df_targets_us = df_targets[df_targets["country"] == "US"].copy()

    print(f"US Sample S1 size: {len(df_s1_us)}, US targets: {len(df_targets_us)}", flush=True)

    s1_name_addr = (df_s1_us["business_name"].fillna('') + " " + df_s1_us["business_address"].fillna('')).values
    target_name_addr = (df_targets_us["business_name"].fillna('') + " " + df_targets_us["business_address"].fillna('')).values
    
    s1_ids = df_s1_us["entity_id"].values
    target_ids = df_targets_us["entity_id"].values

    # Test A: Char_wb (3,5) TF-IDF max_df=0.01
    print("\n--- Test A: Char_wb (3,5) TF-IDF (max_df=0.01, top_k=20) ---", flush=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_df=0.01, min_df=2, max_features=150000, sublinear_tf=True)
    vec_char.fit(target_name_addr[:100000])

    X_target_char = vec_char.transform(target_name_addr)
    X_s1_char = vec_char.transform(s1_name_addr)

    candidates_map_char = {}
    batch_size = 2000
    for start_idx in range(0, X_s1_char.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X_s1_char.shape[0])
        batch_s1 = X_s1_char[start_idx:end_idx]
        sim_matrix = batch_s1.dot(X_target_char.T).tocsr()

        indptr, indices, data = sim_matrix.indptr, sim_matrix.indices, sim_matrix.data
        for i in range(batch_s1.shape[0]):
            s1_id = s1_ids[start_idx + i]
            r_start, r_end = indptr[i], indptr[i+1]
            if r_start == r_end:
                candidates_map_char[s1_id] = []
                continue
            r_data = data[r_start:r_end]
            r_indices = indices[r_start:r_end]
            top_k = min(20, len(r_indices))
            sorted_idx = np.argsort(r_data)[::-1][:top_k]
            candidates_map_char[s1_id] = [target_ids[r_indices[k]] for k in sorted_idx]

    evaluate_recall(candidates_map_char, gt_map)
