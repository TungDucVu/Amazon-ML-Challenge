import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

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
    cand_lens = [len(cands) for cands in candidates_map.values()]
    avg_cands = np.mean(cand_lens)
    p95_cands = np.percentile(cand_lens, 95)
    max_cands = np.max(cand_lens)
    print(f"--> Target-Level Recall: {recall:.2f}% | Mean: {avg_cands:.2f} | P95: {p95_cands:.1f} | Max: {max_cands}", flush=True)
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

    # Sample 10,000 S1 queries
    np.random.seed(42)
    sample_s1_ids = set(np.random.choice(df_s1["entity_id"].values, size=10000, replace=False))
    df_s1_sub = df_s1[df_s1["entity_id"].isin(sample_s1_ids)].copy()

    gt_map = {
        row["source1_entity_id"]: [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        if row["matched_entity_ids"] else []
        for _, row in df_gt.iterrows()
        if row["source1_entity_id"] in sample_s1_ids
    }

    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    df_s1_us = df_s1_sub[df_s1_sub["country"] == "US"].copy()
    df_targets_us = df_targets[df_targets["country"] == "US"].copy()

    print(f"US Sample S1 size: {len(df_s1_us):,}, US targets: {len(df_targets_us):,}", flush=True)

    s1_name_addr = (df_s1_us["business_name"].fillna('') + " " + df_s1_us["business_address"].fillna('')).values
    target_name_addr = (df_targets_us["business_name"].fillna('') + " " + df_targets_us["business_address"].fillna('')).values

    s1_ids = df_s1_us["entity_id"].values
    target_ids = df_targets_us["entity_id"].values

    for max_df_val, min_df_val, top_k_val in [(0.001, 2, 16), (0.0015, 2, 16), (0.001, 2, 20)]:
        print(f"\n--- Benchmark: max_df={max_df_val}, min_df={min_df_val}, top_k={top_k_val} ---", flush=True)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=max_df_val,
            min_df=min_df_val,
            max_features=120000,
            sublinear_tf=True
        )
        vectorizer.fit(target_name_addr[:200000])

        chunk_size = 500000
        X_target = vstack([vectorizer.transform(target_name_addr[i:i+chunk_size]) for i in range(0, len(target_name_addr), chunk_size)], format="csr")
        X_s1 = vectorizer.transform(s1_name_addr)

        candidates_map = {}
        batch_size = 2000
        for start_idx in range(0, X_s1.shape[0], batch_size):
            sim_matrix = X_s1[start_idx:start_idx+batch_size].dot(X_target.T).tocsr()
            indptr, indices, data = sim_matrix.indptr, sim_matrix.indices, sim_matrix.data
            for i in range(sim_matrix.shape[0]):
                s1_id = s1_ids[start_idx + i]
                r_start, r_end = indptr[i], indptr[i+1]
                if r_start == r_end:
                    candidates_map[s1_id] = []
                    continue
                r_data, r_indices = data[r_start:r_end], indices[r_start:r_end]
                top_k = min(top_k_val, len(r_indices))
                sorted_idx = np.argsort(r_data)[::-1][:top_k]
                candidates_map[s1_id] = [target_ids[r_indices[k]] for k in sorted_idx]

        evaluate_recall(candidates_map, gt_map)
