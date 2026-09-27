import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import faiss
import torch
from sparse_dot_topn import sp_matmul_topn
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed

torch.set_num_threads(os.cpu_count() or 4)

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from data_loader import load_source_file, load_ground_truth

DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def parallel_tfidf_transform(vec, texts, chunk_size=150000, n_jobs=-1):
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(vec.transform)(chunk) for chunk in chunks
    )
    return sp.vstack(blocks, format="csr")

def fast_chunked_sparse_topk(X_s1, X_target, target_ids, top_k=15, chunk_size=500000, n_threads=4):
    n_queries = X_s1.shape[0]
    n_targets = X_target.shape[0]

    cand_indices = [[] for _ in range(n_queries)]
    cand_sims = [[] for _ in range(n_queries)]

    for c_start in range(0, n_targets, chunk_size):
        c_end = min(c_start + chunk_size, n_targets)
        X_target_chunk = X_target[c_start:c_end]

        sim_mat = sp_matmul_topn(
            A=X_s1,
            B=X_target_chunk.T,
            top_n=top_k,
            threshold=0.01,
            sort=True,
            n_threads=n_threads
        )

        indptr = sim_mat.indptr
        indices = sim_mat.indices
        data = sim_mat.data

        for q_idx in range(n_queries):
            r_start, r_end = indptr[q_idx], indptr[q_idx + 1]
            if r_start < r_end:
                t_idx_local = indices[r_start:r_end]
                sim_values = data[r_start:r_end]

                t_idx_global = t_idx_local + c_start
                cand_indices[q_idx].extend(t_idx_global)
                cand_sims[q_idx].extend(sim_values)

    results = []
    for q_idx in range(n_queries):
        if not cand_sims[q_idx]:
            results.append([])
        else:
            sims = np.array(cand_sims[q_idx])
            idxs = np.array(cand_indices[q_idx])

            top_k_idx = np.argsort(sims)[::-1][:top_k]
            top_target_ids = [target_ids[idxs[k]] for k in top_k_idx]
            results.append(top_target_ids)

    return results

def get_dense_embeddings(model, texts, batch_size=256):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)

def retrieve_dense_faiss_topk(s1_emb, target_emb, target_ids, top_k=15):
    dim = target_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(target_emb)

    sims, indices = index.search(s1_emb, top_k)
    results = []
    for i in range(len(s1_emb)):
        row_indices = indices[i]
        valid_ids = [target_ids[idx] for idx in row_indices if idx >= 0]
        results.append(valid_ids)
    return results

if __name__ == "__main__":
    val_s1_path = os.path.abspath("dataset/split/val_split/val_source1.tsv")
    val_gt_path = os.path.abspath("dataset/split/val_split/val_ground_truth.tsv")
    s2_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv")
    s3_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv")

    print("Loading Phase 2B benchmarking dataset...", flush=True)
    df_s1 = load_source_file(val_s1_path)
    df_gt = load_ground_truth(val_gt_path)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)

    # Sample 5,000 S1 queries
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

    df_s1_us = df_s1_sub[df_s1_sub["country"] == "US"].copy()
    df_targets_us = df_targets[df_targets["country"] == "US"].copy()

    print(f"US Benchmark S1: {len(df_s1_us):,}, US Targets: {len(df_targets_us):,}", flush=True)

    s1_ids = df_s1_us["entity_id"].values
    target_ids = df_targets_us["entity_id"].values

    s1_name_addr = (df_s1_us["business_name"].fillna('') + " " + df_s1_us["business_address"].fillna('')).values
    target_name_addr = (df_targets_us["business_name"].fillna('') + " " + df_targets_us["business_address"].fillna('')).values

    s1_name = df_s1_us["business_name"].fillna('').values
    target_name = df_targets_us["business_name"].fillna('').values

    n_threads = max(1, (os.cpu_count() or 4) - 1)

    # 1. Expanded Channel 1 (Word Name+Addr: Top 20)
    print("Fitting & Retrieving Channel 1 (Word Name+Addr Top 20)...", flush=True)
    vec1 = TfidfVectorizer(ngram_range=(1,2), max_df=0.005, min_df=2, max_features=80000, sublinear_tf=True)
    vec1.fit(target_name_addr[:200000])
    X_target_1 = parallel_tfidf_transform(vec1, target_name_addr, n_jobs=n_threads)
    X_s1_1 = parallel_tfidf_transform(vec1, s1_name_addr, n_jobs=n_threads)
    ch1_cands = fast_chunked_sparse_topk(X_s1_1, X_target_1, target_ids, top_k=20, n_threads=n_threads)

    # 2. Expanded Channel 2 (Word Name Only: Top 15)
    print("Fitting & Retrieving Channel 2 (Word Name Only Top 15)...", flush=True)
    vec2 = TfidfVectorizer(ngram_range=(1,2), max_df=0.005, min_df=2, max_features=80000, sublinear_tf=True)
    vec2.fit(target_name[:200000])
    X_target_2 = parallel_tfidf_transform(vec2, target_name, n_jobs=n_threads)
    X_s1_2 = parallel_tfidf_transform(vec2, s1_name, n_jobs=n_threads)
    ch2_cands = fast_chunked_sparse_topk(X_s1_2, X_target_2, target_ids, top_k=15, n_threads=n_threads)

    # 3. Expanded Channel 3 (Char_wb Name Only: Top 15)
    print("Fitting & Retrieving Channel 3 (Char_wb Name Only Top 15)...", flush=True)
    vec3 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,4), max_df=0.005, min_df=3, max_features=80000, sublinear_tf=True)
    vec3.fit(target_name[:200000])
    X_target_3 = parallel_tfidf_transform(vec3, target_name, n_jobs=n_threads)
    X_s1_3 = parallel_tfidf_transform(vec3, s1_name, n_jobs=n_threads)
    ch3_cands = fast_chunked_sparse_topk(X_s1_3, X_target_3, target_ids, top_k=15, n_threads=n_threads)

    # 4. Dense Vector Channel (Multilingual MiniLM Top 15 on Target sample)
    print("Loading Dense Multilingual Embedding Model...", flush=True)
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)

    target_sample_size = min(30000, len(target_name_addr))
    print(f"Generating Query & Target Sample ({target_sample_size:,}) Dense Embeddings...", flush=True)
    s1_formatted = [f"Country: US | Name: {n} | Address: {a}" for n, a in zip(s1_name, df_s1_us["business_address"].fillna(''))]
    s1_dense_emb = get_dense_embeddings(dense_model, s1_formatted, batch_size=256)

    target_sample_formatted = [f"Country: US | Name: {n} | Address: {a}" for n, a in zip(target_name[:target_sample_size], df_targets_us["business_address"].fillna('')[:target_sample_size])]
    target_dense_emb = get_dense_embeddings(dense_model, target_sample_formatted, batch_size=256)

    ch4_dense_cands = retrieve_dense_faiss_topk(s1_dense_emb, target_dense_emb, target_ids[:target_sample_size], top_k=15)

    # Tracing Pareto Curve across K in [10, 15, 20, 25, 30, 40]
    print("\n================================================================================", flush=True)
    print("PHASE 2B EMPIRICAL PARETO CURVE RECALL AUDIT (K in [10, 15, 20, 25, 30, 40])", flush=True)
    print("================================================================================\n", flush=True)

    k_values = [10, 15, 20, 25, 30, 40]

    for k_cap in k_values:
        total_true = 0
        retrieved_true = 0
        cand_counts = []

        for i in range(len(s1_ids)):
            s1_id = s1_ids[i]
            combined = ch1_cands[i] + ch2_cands[i] + ch3_cands[i] + ch4_dense_cands[i]
            dedup_cands = list(dict.fromkeys(combined))[:k_cap]
            cand_counts.append(len(dedup_cands))

            true_set = set(gt_map.get(s1_id, []))
            total_true += len(true_set)
            if true_set:
                retrieved_true += len(true_set.intersection(set(dedup_cands)))

        target_recall = (retrieved_true / total_true * 100) if total_true > 0 else 0
        avg_c = np.mean(cand_counts)
        p95_c = np.percentile(cand_counts, 95)
        max_c = np.max(cand_counts)

        print(f"Cap K = {k_cap:>2d} | Target Recall: {target_recall:.2f}% | Mean Cands: {avg_c:.2f} | P95: {p95_c:.1f} | Max: {max_c}", flush=True)

    print("\n================================================================================", flush=True)
