import os
import sys
import pandas as pd

# Add src to path if needed
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_source_file, load_ground_truth


def run_eda(data_dir: str):
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on the training dataset.
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    print("================================================================================")
    print("EXPLORATORY DATA ANALYSIS (EDA) - BUSINESS ENTITY RESOLUTION")
    print("================================================================================\n")

    # 1. Ingestion & File Loading
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"Loading training source files from {train_dir}...")
    df_s1 = load_source_file(s1_path)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)
    df_gt = load_ground_truth(gt_path)

    print(f"  Source 1 rows: {len(df_s1):,}")
    print(f"  Source 2 rows: {len(df_s2):,}")
    print(f"  Source 3 rows: {len(df_s3):,}")
    print(f"  Ground Truth rows: {len(df_gt):,}\n")

    # 2. ID Prefix & Uniqueness Verification
    print("--- 1. ID Prefix & Uniqueness Verification ---")
    prefix_s1 = df_s1["entity_id"].str.startswith("S1-").all()
    prefix_s2 = df_s2["entity_id"].str.startswith("S2-").all()
    prefix_s3 = df_s3["entity_id"].str.startswith("S3-").all()
    prefix_gt_s1 = df_gt["source1_entity_id"].str.startswith("S1-").all()

    print(f"  Source 1 IDs all start with 'S1-': {prefix_s1}")
    print(f"  Source 2 IDs all start with 'S2-': {prefix_s2}")
    print(f"  Source 3 IDs all start with 'S3-': {prefix_s3}")
    print(f"  Ground Truth S1 IDs all start with 'S1-': {prefix_gt_s1}")

    dup_s1 = df_s1["entity_id"].duplicated().sum()
    dup_s2 = df_s2["entity_id"].duplicated().sum()
    dup_s3 = df_s3["entity_id"].duplicated().sum()
    dup_gt = df_gt["source1_entity_id"].duplicated().sum()

    print(f"  Duplicate entity_ids in Source 1: {dup_s1}")
    print(f"  Duplicate entity_ids in Source 2: {dup_s2}")
    print(f"  Duplicate entity_ids in Source 3: {dup_s3}")
    print(f"  Duplicate source1_entity_ids in Ground Truth: {dup_gt}")

    assert dup_s1 == 0, "ERROR: Duplicate entity_id in Source 1"
    assert dup_s2 == 0, "ERROR: Duplicate entity_id in Source 2"
    assert dup_s3 == 0, "ERROR: Duplicate entity_id in Source 3"
    assert dup_gt == 0, "ERROR: Duplicate source1_entity_id in Ground Truth"

    # 3. Country Distribution Analysis
    print("\n--- 2. Country Breakdown ---")
    print("Source 1 Countries:")
    print(df_s1["country"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    print("\nSource 2 Countries:")
    print(df_s2["country"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    print("\nSource 3 Countries:")
    print(df_s3["country"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

    # 4. Ground Truth Integrity & Match Cardinality Profiling
    print("\n--- 3. Ground-Truth Match Cardinality & Cross-Check ---")
    s1_id_set = set(df_s1["entity_id"])
    s2_id_set = set(df_s2["entity_id"])
    s3_id_set = set(df_s3["entity_id"])
    all_s2_s3 = s2_id_set.union(s3_id_set)

    # Check GT S1 IDs exist in S1
    gt_s1_set = set(df_gt["source1_entity_id"])
    missing_gt_s1 = gt_s1_set - s1_id_set
    print(f"  Ground Truth S1 IDs missing in train_source1.tsv: {len(missing_gt_s1)}")
    assert len(missing_gt_s1) == 0, f"Error: Ground truth references missing S1 IDs: {missing_gt_s1}"

    # Parse matches per S1 entity
    match_counts = []
    s2_matches_per_s1 = []
    s3_matches_per_s1 = []
    multi_same_source_count = 0
    missing_target_ids = set()

    for _, row in df_gt.iterrows():
        match_str = row["matched_entity_ids"]
        if not match_str or not match_str.strip():
            matched_ids = []
        else:
            matched_ids = [x.strip() for x in match_str.split(",") if x.strip()]

        # Check all matched IDs exist in S2 or S3
        for m_id in matched_ids:
            if m_id not in all_s2_s3:
                missing_target_ids.add(m_id)

        s2_in_match = [x for x in matched_ids if x.startswith("S2-")]
        s3_in_match = [x for x in matched_ids if x.startswith("S3-")]

        if len(s2_in_match) > 1 or len(s3_in_match) > 1:
            multi_same_source_count += 1

        match_counts.append(len(matched_ids))
        s2_matches_per_s1.append(len(s2_in_match))
        s3_matches_per_s1.append(len(s3_in_match))

    print(f"  Ground Truth target IDs missing in train_source2/source3: {len(missing_target_ids)}")
    assert len(missing_target_ids) == 0, f"Error: Ground truth references missing target IDs: {missing_target_ids}"

    df_gt["match_count"] = match_counts
    print("\nGround Truth Match Count Distribution:")
    cardinality_counts = df_gt["match_count"].value_counts().sort_index()
    cardinality_pcts = (cardinality_counts / len(df_gt) * 100).round(2)
    for c_val, count in cardinality_counts.items():
        print(f"  {c_val} matches: {count:,} entities ({cardinality_pcts[c_val]}%)")

    zero_match_count = (df_gt["match_count"] == 0).sum()
    one_match_count = (df_gt["match_count"] == 1).sum()
    multi_match_count = (df_gt["match_count"] > 1).sum()

    print(f"\nSummary Breakdown:")
    print(f"  Singletons (0 matches): {zero_match_count:,} ({zero_match_count/len(df_gt)*100:.2f}%)")
    print(f"  1-to-1 Matches (1 match): {one_match_count:,} ({one_match_count/len(df_gt)*100:.2f}%)")
    print(f"  1-to-Many Matches (>=2 matches): {multi_match_count:,} ({multi_match_count/len(df_gt)*100:.2f}%)")
    print(f"  S1 entities matching MULTIPLE records within SAME source file: {multi_same_source_count:,} ({multi_same_source_count/len(df_gt)*100:.2f}%)")

    # 5. Null & String Statistics
    print("\n--- 4. String & Field Quality Check ---")
    for name, df in [("Source 1", df_s1), ("Source 2", df_s2), ("Source 3", df_s3)]:
        empty_names = (df["business_name"].str.strip() == "").sum()
        empty_addrs = (df["business_address"].str.strip() == "").sum()
        print(f"  {name}: Empty names = {empty_names}, Empty addresses = {empty_addrs}")

    print("\n================================================================================")
    print("EDA COMPLETED SUCCESSFULLY - ALL INTEGRITY ASSERTS PASSED")
    print("================================================================================\n")


if __name__ == "__main__":
    default_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "6ab10eb3b23ba_student_resource", "student_resource", "dataset"
    )
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    run_eda(os.path.abspath(data_dir))
