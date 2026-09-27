import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from data_loader import load_source_file, load_ground_truth

def fast_chunked_topk(X_s1, X_target, target_ids, top_k=10, chunk_size=500000):
    """
    Computes top-K candidates across X_target in chunks using sparse_dot_topn.
    Returns: dict mapping query row index -> list of target entity IDs.
    """
    n_queries = X_s1.shape[0]
    n_targets = X_target.shape[0]
    
    # Store top candidate index & sim per query row
    cand_indices = [[] for _ in range(n_queries)]
    cand_sims = [[] for _ in range(n_queries)]
    
    for c_start in range(0, n_targets, chunk_size):
        c_end = min(c_start + chunk_size, n_targets)
        X_target_chunk = X_target[c_start:c_end]
        
        # C++ top-n matrix multiplication
        sim_mat = sp_matmul_topn(
            A=X_s1,
            B=X_target_chunk.T,
            top_n=top_k,
            lower_bound=0.01
        )
        
        indptr = sim_mat.indptr
        indices = sim_mat.indices
        data = sim_mat.data
        
        for q_idx in range(n_queries):
            r_start, r_end = indptr[q_idx], indptr[q_idx + 1]
            if r_start < r_end:
                t_idx_local = indices[r_start:r_end]
                sim_values = data[r_start:r_end]
                
                # Map chunk-local indices to global target indices
                t_idx_global = t_idx_local + c_start
                cand_indices[q_idx].extend(t_idx_global)
                cand_sims[q_idx].extend(sim_values)
                
    # Select final top-K per query row across all chunks
    results = []
    for q_idx in range(n_queries):
        if not cand_sims[q_idx]:
            results.append([])
        else:
            sims = np.array(cand_sims[q_idx])
            idxs = np.array(cand_indices[q_idx])
            
            top_k_indices = np.argsort(sims)[::-1][:top_k]
            top_target_ids = [target_ids[idxs[k]] for k in top_k_indices]
            results.append(top_target_ids)
            
    return results

if __name__ == "__main__":
    val_s1_path = os.path.abspath("dataset/split/val_split/val_source1.tsv")
    val_gt_path = os.path.abspath("dataset/split/val_split/val_ground_truth.tsv")
    s2_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv")
    s3_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv")

    print("Loading data...", flush=True)
    df_s1 = load_source_file(val_s1_path)
    df_gt = load_ground_truth(val_gt_path)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)

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

    print(f"US Sample S1: {len(df_s1_us):,}, US targets: {len(df_targets_us):,}", flush=True)

    s1_ids = df_s1_us["entity_id"].values
    target_ids = df_targets_us["entity_id"].values

    s1_name_addr = (df_s1_us["business_name"].fillna('') + " " + df_s1_us["business_address"].fillna('')).values
    target_name_addr = (df_targets_us["business_name"].fillna('') + " " + df_targets_us["business_address"].fillna('')).values

    s1_name = df_s1_us["business_name"].fillna('').values
    target_name = df_targets_us["business_name"].fillna('').values

    t0 = time.time()

    # Channel 1: Word TF-IDF Name+Address
    print("Fitting Channel 1 (Word TF-IDF Name+Addr)...", flush=True)
    vec1 = TfidfVectorizer(ngram_range=(1,2), max_df=0.005, min_df=2, max_features=80000, sublinear_tf=True)
    vec1.fit(target_name_addr[:200000])

    print("Transforming Target and Query matrices...", flush=True)
    X_target_1 = sp.vstack([vec1.transform(target_name_addr[i:i+500000]) for i in range(0, len(target_name_addr), 500000)], format="csr")
    X_s1_1 = vec1.transform(s1_name_addr)

    print("Retrieving Channel 1 candidates with sparse_dot_topn...", flush=True)
    ch1_cands = fast_chunked_topk(X_s1_1, X_target_1, target_ids, top_k=10)

    # Channel 2: Word TF-IDF Name-Only
    print("Fitting Channel 2 (Word TF-IDF Name-Only)...", flush=True)
    vec2 = TfidfVectorizer(ngram_range=(1,2), max_df=0.005, min_df=2, max_features=80000, sublinear_tf=True)
    vec2.fit(target_name[:200000])

    X_target_2 = sp.vstack([vec2.transform(target_name[i:i+500000]) for i in range(0, len(target_name), 500000)], format="csr")
    X_s1_2 = vec2.transform(s1_name)

    print("Retrieving Channel 2 candidates with sparse_dot_topn...", flush=True)
    ch2_cands = fast_chunked_topk(X_s1_2, X_target_2, target_ids, top_k=8)

    t1 = time.time()
    print(f"Elapsed blocking time for 10,000 queries vs 4.5M targets: {t1 - t0:.2f} seconds!", flush=True)

    # Combine & Evaluate Recall
    total_true = 0
    retrieved_true = 0
    cand_counts = []

    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        cands = list(dict.fromkeys(ch1_cands[i] + ch2_cands[i]))[:20]
        cand_counts.append(len(cands))
        
        true_set = set(gt_map.get(s1_id, []))
        total_true += len(true_set)
        if true_set:
            retrieved_true += len(true_set.intersection(set(cands)))

    recall = (retrieved_true / total_true * 100) if total_true > 0 else 0
    avg_cands = np.mean(cand_counts)
    p95_cands = np.percentile(cand_counts, 95)
    max_cands = np.max(cand_counts)

    print(f"--> Target Recall: {recall:.2f}% | Avg Candidates: {avg_cands:.2f} | P95: {p95_cands:.1f} | Max: {max_cands}", flush=True)
