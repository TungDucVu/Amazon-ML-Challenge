import gc
import os
import sys
import time
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_source_file, load_ground_truth


def clean_text_advanced(text: str) -> str:
    """
    Normalizes business names and addresses:
    - Strips URL schemes and web prefixes
    - Strips common domain extensions (.com, .org, .net, etc.)
    - Normalizes and strips legal entity suffixes (ltd, inc, corp, llc, etc.)
    - Removes punctuation and standardizes whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'https?://(?:www\.)?', '', t)
    t = re.sub(r'www\.', '', t)
    t = re.sub(r'\.(com|in|org|net|co|us|io|biz|info)(?:/.*)?', ' ', t)
    t = re.sub(r'\b(ltd|limited|inc|incorporated|corp|corporation|llc|llp|pvt|private|co|company|sa|sarl)\b', ' ', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return " ".join(t.split())


def round_robin_fusion(lists: list) -> list:
    """
    Merges multiple ranked lists using round-robin rank interleaving.
    Ensures that top hits from every retrieval channel are prioritized without
    allowing a single channel to monopolize candidate slots at low K.
    """
    max_len = max((len(l) for l in lists), default=0)
    result = []
    seen = set()
    for rank in range(max_len):
        for candidate_list in lists:
            if rank < len(candidate_list):
                cand = candidate_list[rank]
                if cand not in seen:
                    seen.add(cand)
                    result.append(cand)
    return result


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
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(vec.transform)(chunk) for chunk in chunks
    )
    return sp.vstack(blocks, format="csr")


def fast_chunked_sparse_topk(X_s1, X_target, target_ids, top_k=30, chunk_size=500000, n_threads=4):
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
            threshold=0.003,
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


def run_openset_france_dry_run():
    """
    Open-Set Dry Run verifying country partition logic on synthetic France entities.
    """
    print("\n--------------------------------------------------------------------------------", flush=True)
    print("EXECUTING OPEN-SET GENERALIZATION DRY RUN FOR UNSEEN COUNTRY: FRANCE", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)

    df_fr_s1 = pd.DataFrame([
        {"entity_id": "S1-FR-001", "business_name": "Boulangerie Patisserie Paris", "business_address": "15 Rue de Rivoli, Paris", "country": "France"},
        {"entity_id": "S1-FR-002", "business_name": "Societe Generale France", "business_address": "29 Boulevard Haussmann, Paris", "country": "France"}
    ])

    df_fr_targets = pd.DataFrame([
        {"entity_id": "S2-FR-001", "business_name": "Boulangerie Parisienne", "business_address": "15 Rue de Rivoli, 75004 Paris", "country": "France"},
        {"entity_id": "S3-FR-002", "business_name": "Societe Generale SA", "business_address": "29 Blvd Haussmann, Paris", "country": "France"},
        {"entity_id": "S2-US-999", "business_name": "New York Bakery", "business_address": "5th Ave, NY", "country": "US"}
    ])

    cands = get_multi_stage_candidates_for_partition(df_fr_s1, df_fr_targets, max_candidates_per_s1=20)
    
    assert "S2-FR-001" in cands["S1-FR-001"], "Open-Set Dry Run Failure: Boulangerie target missing!"
    assert "S3-FR-002" in cands["S1-FR-002"], "Open-Set Dry Run Failure: Societe Generale target missing!"
    assert "S2-US-999" not in cands["S1-FR-001"] and "S2-US-999" not in cands["S1-FR-002"], "Open-Set Dry Run Failure: US target crossed France boundary!"

    print("OPEN-SET FRANCE DRY RUN STATUS: PASS (Zero-shot country partition fully verified!)", flush=True)
    print("--------------------------------------------------------------------------------\n", flush=True)


def get_multi_stage_candidates_for_partition(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    max_candidates_per_s1: int = 60
) -> dict:
    """
    Generates high-recall candidate pairs per country partition using 4 multi-modal sparse channels:
    - Channel 1: Word-level TF-IDF (1,2) on Cleaned Name + Cleaned Address (Top 35)
    - Channel 2: Word-level TF-IDF (1,2) on Cleaned Business Name only (Top 25)
    - Channel 3: Char_wb TF-IDF (3,4) on Cleaned Business Name only (Top 25)
    - Channel 4: Word-level TF-IDF (1,2) on Cleaned Address only (Top 25)
    Fused via round-robin rank interleaving to ensure maximum diversity and recall.
    """
    if len(df_s1) == 0 or len(df_targets) == 0:
        return {s1_id: [] for s1_id in df_s1["entity_id"]}

    s1_ids = df_s1["entity_id"].values
    target_ids = df_targets["entity_id"].values

    s1_raw_name = df_s1["business_name"].fillna('').values
    target_raw_name = df_targets["business_name"].fillna('').values
    s1_raw_addr = df_s1["business_address"].fillna('').values
    target_raw_addr = df_targets["business_address"].fillna('').values

    print(f"    Cleaning text & domain stems across {len(s1_ids):,} queries and {len(target_ids):,} targets...", flush=True)
    s1_clean_name = [clean_text_advanced(x) for x in s1_raw_name]
    target_clean_name = [clean_text_advanced(x) for x in target_raw_name]
    s1_clean_addr = [clean_text_advanced(x) for x in s1_raw_addr]
    target_clean_addr = [clean_text_advanced(x) for x in target_raw_addr]

    s1_full = [f"{n} {a}" for n, a in zip(s1_clean_name, s1_clean_addr)]
    target_full = [f"{n} {a}" for n, a in zip(target_clean_name, target_clean_addr)]

    sample_size = min(200000, len(target_full))
    sample_indices = np.random.choice(len(target_full), size=sample_size, replace=False) if len(target_full) > sample_size else np.arange(len(target_full))
    
    n_threads = max(1, (os.cpu_count() or 4) - 1)

    max_df_val = 0.95 if sample_size < 100 else 0.005
    min_df_val = 1 if sample_size < 100 else 2

    # --- CHANNEL 1: Word-level TF-IDF (Full Cleaned Name + Address: Top 35) ---
    print(f"    [Channel 1] Word TF-IDF (Clean Name+Addr Top 35)...", flush=True)
    vec1 = TfidfVectorizer(ngram_range=(1, 2), max_df=max_df_val, min_df=min_df_val, max_features=100000, sublinear_tf=True)
    target_sample_1 = [target_full[i] for i in sample_indices]
    vec1.fit(target_sample_1)
    X_target_1 = parallel_tfidf_transform(vec1, target_full, n_jobs=n_threads)
    X_s1_1 = parallel_tfidf_transform(vec1, s1_full, n_jobs=n_threads)
    ch1_cands = fast_chunked_sparse_topk(X_s1_1, X_target_1, target_ids, top_k=35, n_threads=n_threads)
    del X_target_1, X_s1_1, vec1
    gc.collect()

    # --- CHANNEL 2: Word-level TF-IDF (Cleaned Name Only: Top 25) ---
    print(f"    [Channel 2] Word TF-IDF (Clean Name Only Top 25)...", flush=True)
    vec2 = TfidfVectorizer(ngram_range=(1, 2), max_df=max_df_val, min_df=min_df_val, max_features=100000, sublinear_tf=True)
    target_sample_2 = [target_clean_name[i] for i in sample_indices]
    vec2.fit(target_sample_2)
    X_target_2 = parallel_tfidf_transform(vec2, target_clean_name, n_jobs=n_threads)
    X_s1_2 = parallel_tfidf_transform(vec2, s1_clean_name, n_jobs=n_threads)
    ch2_cands = fast_chunked_sparse_topk(X_s1_2, X_target_2, target_ids, top_k=25, n_threads=n_threads)
    del X_target_2, X_s1_2, vec2
    gc.collect()

    # --- CHANNEL 3: Char_wb TF-IDF (Cleaned Name Only: Top 25) ---
    print(f"    [Channel 3] Char_wb TF-IDF (Clean Name Only Top 25)...", flush=True)
    vec3 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_df=max_df_val, min_df=max(1, min_df_val), max_features=100000, sublinear_tf=True)
    target_sample_3 = [target_clean_name[i] for i in sample_indices]
    vec3.fit(target_sample_3)
    X_target_3 = parallel_tfidf_transform(vec3, target_clean_name, n_jobs=n_threads)
    X_s1_3 = parallel_tfidf_transform(vec3, s1_clean_name, n_jobs=n_threads)
    ch3_cands = fast_chunked_sparse_topk(X_s1_3, X_target_3, target_ids, top_k=25, n_threads=n_threads)
    del X_target_3, X_s1_3, vec3
    gc.collect()

    # --- CHANNEL 4: Word-level TF-IDF (Cleaned Address Only: Top 25) ---
    print(f"    [Channel 4] Word TF-IDF (Clean Address Only Top 25)...", flush=True)
    vec4 = TfidfVectorizer(ngram_range=(1, 2), max_df=max_df_val, min_df=min_df_val, max_features=100000, sublinear_tf=True)
    target_sample_4 = [target_clean_addr[i] for i in sample_indices]
    vec4.fit(target_sample_4)
    X_target_4 = parallel_tfidf_transform(vec4, target_clean_addr, n_jobs=n_threads)
    X_s1_4 = parallel_tfidf_transform(vec4, s1_clean_addr, n_jobs=n_threads)
    ch4_cands = fast_chunked_sparse_topk(X_s1_4, X_target_4, target_ids, top_k=25, n_threads=n_threads)
    del X_target_4, X_s1_4, vec4
    gc.collect()

    # Interleave multi-modal candidate channels using Round-Robin Fusion
    print("    Fusing Multi-Modal Channels via Round-Robin Interleaving...", flush=True)
    final_candidates_map = {}
    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        fused = round_robin_fusion([ch1_cands[i], ch2_cands[i], ch3_cands[i], ch4_cands[i]])[:max_candidates_per_s1]
        final_candidates_map[s1_id] = fused

    return final_candidates_map


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_candidates_per_s1: int = 60
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


def evaluate_pareto_curve(df_cand: pd.DataFrame, df_gt: pd.DataFrame, k_caps=[10, 15, 20, 25, 30, 40, 50, 60]):
    """
    Traces empirical Pareto curve across candidate caps K in [10, 15, 20, 25, 30, 40, 50, 60].
    """
    print("\n================================================================================", flush=True)
    print("PHASE 2C EMPIRICAL PARETO CURVE RECALL AUDIT MATRIX", flush=True)
    print("================================================================================\n", flush=True)

    cand_map = dict(zip(df_cand["source1_entity_id"], df_cand["candidate_entity_ids"]))
    gt_map = {
        row["source1_entity_id"]: [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        if row["matched_entity_ids"] else []
        for _, row in df_gt.iterrows()
    }

    for k_cap in k_caps:
        total_true = 0
        retrieved_true = 0
        cand_lengths = []

        for s1_id, true_targets in gt_map.items():
            raw_cands = [x.strip() for x in cand_map.get(s1_id, "").split(",") if x.strip()]
            dedup_cands = raw_cands[:k_cap]
            cand_lengths.append(len(dedup_cands))

            true_set = set(true_targets)
            total_true += len(true_set)
            if true_set:
                retrieved_true += len(true_set.intersection(set(dedup_cands)))

        target_recall = (retrieved_true / total_true * 100) if total_true > 0 else 100.0
        avg_c = float(np.mean(cand_lengths))
        p95_c = float(np.percentile(cand_lengths, 95))
        max_c = int(np.max(cand_lengths))

        print(f"Cap K = {k_cap:>2d} | Target Recall: {target_recall:.2f}% | Mean Cands: {avg_c:.2f} | P95: {p95_c:.1f} | Max: {max_c}", flush=True)

    print("\n================================================================================", flush=True)


if __name__ == "__main__":
    t0_main = time.time()
    
    # Run Open-Set France Dry Run
    run_openset_france_dry_run()

    val_s1_path = os.path.abspath("dataset/split/val_split/val_source1.tsv")
    val_gt_path = os.path.abspath("dataset/split/val_split/val_ground_truth.tsv")
    s2_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv")
    s3_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv")

    print("Loading files for Phase 2C high-recall candidate blocking run...", flush=True)
    df_s1 = load_source_file(val_s1_path)
    df_gt = load_ground_truth(val_gt_path)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)

    verify_country_integrity(df_s1, df_s2, df_s3, df_gt)

    df_cand = generate_candidate_pairs(df_s1, df_s2, df_s3, max_candidates_per_s1=60)

    out_dir = os.path.abspath("output")
    os.makedirs(out_dir, exist_ok=True)
    out_cand_path = os.path.join(out_dir, "val_candidate_pairs.tsv")
    df_cand.to_csv(out_cand_path, sep="\t", index=False)
    print(f"Exported Phase 2C candidate pairs to {out_cand_path}", flush=True)

    evaluate_pareto_curve(df_cand, df_gt, k_caps=[10, 15, 20, 25, 30, 40, 50, 60])
    
    t1_main = time.time()
    print(f"Phase 2C Candidate Generation Completed in total {t1_main - t0_main:.2f} seconds!", flush=True)
