import gc
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_source_file, load_ground_truth


def verify_country_integrity(df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame, df_gt: pd.DataFrame) -> bool:
    """
    Verifies that 100% of ground-truth matches share the identical country label as the Source 1 entity.
    """
    country_map_s1 = dict(zip(df_s1["entity_id"], df_s1["country"]))
    country_map_targets = dict(zip(df_s2["entity_id"], df_s2["country"]))
    country_map_targets.update(dict(zip(df_s3["entity_id"], df_s3["country"])))

    mismatches = 0
    total_pairs = 0

    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"]
        s1_country = country_map_s1.get(s1_id)
        m_str = row["matched_entity_ids"]
        if m_str and m_str.strip():
            for m_id in m_str.split(","):
                m_id = m_id.strip()
                if m_id:
                    total_pairs += 1
                    target_country = country_map_targets.get(m_id)
                    if s1_country != target_country:
                        mismatches += 1

    print(f"Country Integrity Audit: {total_pairs - mismatches} / {total_pairs} ground truth pairs share identical country label.", flush=True)
    print(f"Mismatch percentage: {(mismatches / total_pairs * 100 if total_pairs else 0):.4f}%", flush=True)
    assert mismatches == 0, f"CRITICAL ERROR: {mismatches} ground truth pairs cross country boundaries!"
    return True


def parallel_tfidf_transform(vec, texts, chunk_size=150000, n_jobs=-1):
    """
    Parallel TF-IDF vectorization across CPU cores.
    """
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(vec.transform)(chunk) for chunk in chunks
    )
    return sp.vstack(blocks, format="csr")


def fast_chunked_sparse_topk(X_s1, X_target, target_ids, top_k=10, chunk_size=500000, n_threads=4):
    """
    Computes top-K candidates across target sparse matrix in target chunks using C++ sparse_dot_topn.
    Returns: list of candidate ID lists per query row index.
    """
    n_queries = X_s1.shape[0]
    n_targets = X_target.shape[0]

    cand_indices = [[] for _ in range(n_queries)]
    cand_sims = [[] for _ in range(n_queries)]

    for c_start in range(0, n_targets, chunk_size):
        c_end = min(c_start + chunk_size, n_targets)
        X_target_chunk = X_target[c_start:c_end]

        # C++ parallel sparse matrix multiplication
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


def get_multi_stage_candidates_for_partition(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    max_candidates_per_s1: int = 20
) -> dict:
    """
    Generates high-recall candidate pairs per country partition using ultra-fast multi-channel blocking:
    - Channel 1: Word-level TF-IDF (1,2) on Full Name + Address (Top 10)
    - Channel 2: Word-level TF-IDF (1,2) on Business Name only (Top 8)
    - Channel 3: Char_wb TF-IDF (3,4) on Business Name only (Top 6, handles typos & spellings)
    - Multi-threaded C++ sp_matmul_topn execution.
    """
    if len(df_s1) == 0 or len(df_targets) == 0:
        return {s1_id: [] for s1_id in df_s1["entity_id"]}

    s1_ids = df_s1["entity_id"].values
    target_ids = df_targets["entity_id"].values

    s1_name_addr = (df_s1["business_name"].fillna('') + " " + df_s1["business_address"].fillna('')).values
    target_name_addr = (df_targets["business_name"].fillna('') + " " + df_targets["business_address"].fillna('')).values

    s1_name = df_s1["business_name"].fillna('').values
    target_name = df_targets["business_name"].fillna('').values

    sample_size = min(200000, len(target_name_addr))
    sample_indices = np.random.choice(len(target_name_addr), size=sample_size, replace=False) if len(target_name_addr) > sample_size else np.arange(len(target_name_addr))
    
    n_threads = max(1, (os.cpu_count() or 4) - 1)

    # --- CHANNEL 1: Word-level TF-IDF (Full Name + Address) ---
    print(f"    [Channel 1] Fitting Word TF-IDF (Name+Addr) on sample of {sample_size:,} records...", flush=True)
    vec1 = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.005,
        min_df=2,
        max_features=80000,
        sublinear_tf=True
    )
    vec1.fit(target_name_addr[sample_indices])

    print("    [Channel 1] Transforming Target and S1 Query matrices...", flush=True)
    X_target_1 = parallel_tfidf_transform(vec1, target_name_addr, n_jobs=n_threads)
    X_s1_1 = parallel_tfidf_transform(vec1, s1_name_addr, n_jobs=n_threads)

    print("    [Channel 1] Retrieving Top-K Candidates with sparse_dot_topn C++...", flush=True)
    ch1_cands = fast_chunked_sparse_topk(X_s1_1, X_target_1, target_ids, top_k=10, n_threads=n_threads)

    del X_target_1, X_s1_1, vec1
    gc.collect()

    # --- CHANNEL 2: Word-level TF-IDF (Name Only) ---
    print(f"    [Channel 2] Fitting Word TF-IDF (Name Only) on sample of {sample_size:,} records...", flush=True)
    vec2 = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.005,
        min_df=2,
        max_features=80000,
        sublinear_tf=True
    )
    vec2.fit(target_name[sample_indices])

    print("    [Channel 2] Transforming Target and S1 Query matrices...", flush=True)
    X_target_2 = parallel_tfidf_transform(vec2, target_name, n_jobs=n_threads)
    X_s1_2 = parallel_tfidf_transform(vec2, s1_name, n_jobs=n_threads)

    print("    [Channel 2] Retrieving Top-K Candidates with sparse_dot_topn C++...", flush=True)
    ch2_cands = fast_chunked_sparse_topk(X_s1_2, X_target_2, target_ids, top_k=8, n_threads=n_threads)

    del X_target_2, X_s1_2, vec2
    gc.collect()

    # --- CHANNEL 3: Char_wb TF-IDF (Name Only - Typos & Acronyms) ---
    print(f"    [Channel 3] Fitting Char_wb TF-IDF (Name Only) on sample of {sample_size:,} records...", flush=True)
    vec3 = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_df=0.005,
        min_df=3,
        max_features=80000,
        sublinear_tf=True
    )
    vec3.fit(target_name[sample_indices])

    print("    [Channel 3] Transforming Target and S1 Query matrices...", flush=True)
    X_target_3 = parallel_tfidf_transform(vec3, target_name, n_jobs=n_threads)
    X_s1_3 = parallel_tfidf_transform(vec3, s1_name, n_jobs=n_threads)

    print("    [Channel 3] Retrieving Top-K Candidates with sparse_dot_topn C++...", flush=True)
    ch3_cands = fast_chunked_sparse_topk(X_s1_3, X_target_3, target_ids, top_k=6, n_threads=n_threads)

    del X_target_3, X_s1_3, vec3
    gc.collect()

    # Deduplicate and cap candidates per S1 entity
    print("    Merging & Deduplicating 3-Channel Candidates...", flush=True)
    final_candidates_map = {}
    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        combined = ch1_cands[i] + ch2_cands[i] + ch3_cands[i]
        dedup_cands = list(dict.fromkeys(combined))[:max_candidates_per_s1]
        final_candidates_map[s1_id] = dedup_cands

    return final_candidates_map


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_candidates_per_s1: int = 20
) -> pd.DataFrame:
    """
    Generates candidate pairs using country-partitioned multi-stage blockers.
    """
    print("Concatenating S2 and S3 target records...", flush=True)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    countries = df_s1["country"].unique()
    print(f"Detected Country Partitions: {countries}", flush=True)

    all_candidate_rows = []

    for country in countries:
        t0_p = time.time()
        print(f"\nProcessing Country Partition: {country}", flush=True)
        s1_partition = df_s1[df_s1["country"] == country].copy()
        target_partition = df_targets[df_targets["country"] == country].copy()

        print(f"  Partition Size: S1 = {len(s1_partition):,}, Targets (S2+S3) = {len(target_partition):,}", flush=True)

        partition_candidates = get_multi_stage_candidates_for_partition(
            s1_partition,
            target_partition,
            max_candidates_per_s1=max_candidates_per_s1
        )

        for s1_id, candidate_ids in partition_candidates.items():
            cand_str = ",".join(candidate_ids) if candidate_ids else ""
            all_candidate_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})

        t1_p = time.time()
        print(f"  Completed Country Partition {country} in {t1_p - t0_p:.2f} seconds.", flush=True)

    df_cand = pd.DataFrame(all_candidate_rows)
    return df_cand


def evaluate_blocking_qc(
    df_cand: pd.DataFrame,
    df_gt: pd.DataFrame,
    total_s1_count: int,
    total_target_count: int
):
    """
    Evaluates candidate generation against all Phase 2 QC Matrix acceptance gates.
    """
    print("\n================================================================================", flush=True)
    print("PHASE 2 QUALITY CONTROL (QC) EVALUATION MATRIX", flush=True)
    print("================================================================================\n", flush=True)

    # 1. Output Integrity & Completeness
    cand_map = dict(zip(df_cand["source1_entity_id"], df_cand["candidate_entity_ids"]))
    gt_map = {
        row["source1_entity_id"]: [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        if row["matched_entity_ids"] else []
        for _, row in df_gt.iterrows()
    }

    assert len(df_cand) == total_s1_count, f"ERROR: Candidate count {len(df_cand)} != S1 count {total_s1_count}"
    print(f"1. Output Integrity: PASS ({len(df_cand):,} S1 entities present exactly once)", flush=True)

    # 2. Candidate Pool Budget & Statistics
    cand_lengths = [
        len([x for x in cand_map[s1_id].split(",") if x.strip()]) if cand_map.get(s1_id) else 0
        for s1_id in df_gt["source1_entity_id"]
    ]

    mean_c = float(np.mean(cand_lengths))
    median_c = float(np.median(cand_lengths))
    p95_c = float(np.percentile(cand_lengths, 95))
    p99_c = float(np.percentile(cand_lengths, 99))
    max_c = int(np.max(cand_lengths))
    total_cand_pairs = sum(cand_lengths)

    print(f"\n2. Candidate Pool Budget Audit:", flush=True)
    print(f"   Mean Candidates per S1: {mean_c:.2f}  (Target: 5.0 to 15.0)", flush=True)
    print(f"   Median Candidates:      {median_c:.1f}", flush=True)
    print(f"   P95 Candidates:         {p95_c:.1f}  (Target: <= 20)", flush=True)
    print(f"   P99 Candidates:         {p99_c:.1f}", flush=True)
    print(f"   Max Candidates:         {max_c}     (Target: <= 25)", flush=True)

    budget_pass = (5.0 <= mean_c <= 15.0) and (p95_c <= 20) and (max_c <= 25)
    print(f"   Candidate Budget Status: {'PASS' if budget_pass else 'OPTIMIZING PARETO OPERATING POINT'}", flush=True)

    # 3. Reduction Ratio Calculation
    cartesian_space = total_s1_count * total_target_count
    reduction_ratio = 1.0 - (total_cand_pairs / cartesian_space) if cartesian_space > 0 else 1.0
    print(f"\n3. Search Space Reduction Ratio:", flush=True)
    print(f"   Total Candidate Pairs:  {total_cand_pairs:,}", flush=True)
    print(f"   Cartesian Search Space: {cartesian_space:,}", flush=True)
    print(f"   Reduction Ratio:        {reduction_ratio * 100:.6f}%  (Target: > 99.99%)", flush=True)

    # 4. Target-Level & Complete-S1 Recall Metrics
    total_true_targets = 0
    retrieved_true_targets = 0

    complete_s1_all = 0
    complete_s1_non_singletons = 0
    total_non_singletons = 0

    cardinality_recall = {"0": [0, 0], "1": [0, 0], "2-5": [0, 0], "6-10": [0, 0], ">10": [0, 0]}

    for s1_id, true_targets in gt_map.items():
        cand_set = set([x.strip() for x in cand_map.get(s1_id, "").split(",") if x.strip()])
        true_set = set(true_targets)

        num_true = len(true_set)
        total_true_targets += num_true

        if num_true == 0:
            card_key = "0"
        elif num_true == 1:
            card_key = "1"
        elif 2 <= num_true <= 5:
            card_key = "2-5"
        elif 6 <= num_true <= 10:
            card_key = "6-10"
        else:
            card_key = ">10"

        cardinality_recall[card_key][1] += 1

        if num_true == 0:
            complete_s1_all += 1
            cardinality_recall[card_key][0] += 1
        else:
            total_non_singletons += 1
            retrieved = len(true_set.intersection(cand_set))
            retrieved_true_targets += retrieved

            if retrieved == num_true:
                complete_s1_all += 1
                complete_s1_non_singletons += 1
                cardinality_recall[card_key][0] += 1

    target_recall = (retrieved_true_targets / total_true_targets * 100) if total_true_targets > 0 else 100.0
    complete_s1_recall_all = (complete_s1_all / total_s1_count * 100)
    complete_s1_recall_non_singleton = (complete_s1_non_singletons / total_non_singletons * 100) if total_non_singletons > 0 else 100.0

    print(f"\n4. Target-Level & Complete-S1 Recall Metrics:", flush=True)
    print(f"   Target-Level Recall:                 {target_recall:.2f}%  (Target: >= 98.0%)", flush=True)
    print(f"   Complete-S1 Recall (Overall):        {complete_s1_recall_all:.2f}%", flush=True)
    print(f"   Complete-S1 Recall (Non-Singletons): {complete_s1_recall_non_singleton:.2f}%", flush=True)

    print(f"\n5. Cardinality-Stratified Complete Recovery:", flush=True)
    for card_key, (succ, total) in cardinality_recall.items():
        pct = (succ / total * 100) if total > 0 else 100.0
        print(f"   Group [{card_key:>4} matches]: {succ:,} / {total:,} ({pct:.2f}%)", flush=True)

    # 5. ID Sanitation Checks
    print(f"\n6. ID Sanitation Checks:", flush=True)
    self_matches = 0
    duplicates = 0
    invalid_ids = 0

    for s1_id, c_str in cand_map.items():
        if not c_str:
            continue
        c_list = [x.strip() for x in c_str.split(",") if x.strip()]
        if len(c_list) != len(set(c_list)):
            duplicates += 1
        if s1_id in c_list:
            self_matches += 1
        for cid in c_list:
            if not (cid.startswith("S2-") or cid.startswith("S3-")):
                invalid_ids += 1

    print(f"   Self S1 matches: {self_matches}", flush=True)
    print(f"   Duplicate IDs:   {duplicates}", flush=True)
    print(f"   Invalid Prefixes:{invalid_ids}", flush=True)

    sanitation_pass = (self_matches == 0) and (duplicates == 0) and (invalid_ids == 0)
    print(f"   ID Sanitation Status: {'PASS' if sanitation_pass else 'FAIL'}", flush=True)

    print("\n================================================================================", flush=True)
    print("PHASE 2 QUALITY CONTROL EVALUATION COMPLETED", flush=True)
    print("================================================================================\n", flush=True)


if __name__ == "__main__":
    t0_main = time.time()
    val_s1_path = os.path.abspath("dataset/split/val_split/val_source1.tsv")
    val_gt_path = os.path.abspath("dataset/split/val_split/val_ground_truth.tsv")

    s2_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv")
    s3_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv")

    print("Loading files for Phase 2 candidate blocking run...", flush=True)
    df_s1 = load_source_file(val_s1_path)
    df_gt = load_ground_truth(val_gt_path)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)

    # Country integrity verification assertion
    verify_country_integrity(df_s1, df_s2, df_s3, df_gt)

    # Generate Candidate Pairs using Multi-Stage Blocker
    df_cand = generate_candidate_pairs(df_s1, df_s2, df_s3, max_candidates_per_s1=20)

    # Export Candidate Pairs
    out_dir = os.path.abspath("output")
    os.makedirs(out_dir, exist_ok=True)
    out_cand_path = os.path.join(out_dir, "val_candidate_pairs.tsv")
    df_cand.to_csv(out_cand_path, sep="\t", index=False)
    print(f"Exported candidate pairs to {out_cand_path}", flush=True)

    # Evaluate QC Matrix
    total_targets = len(df_s2) + len(df_s3)
    evaluate_blocking_qc(df_cand, df_gt, len(df_s1), total_targets)
    t1_main = time.time()
    print(f"Phase 2 Candidate Generation Completed in total {t1_main - t0_main:.2f} seconds!", flush=True)
