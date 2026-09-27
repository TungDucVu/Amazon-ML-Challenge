import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_source_file, load_ground_truth

RANDOM_SEED = 42


def create_validation_split(data_dir: str, output_dir: str, val_ratio: float = 0.20):
    """
    Creates a stratified 80/20 train/validation split.
    
    STRICT RULES:
    1. Only train_source1.tsv and train_ground_truth.tsv are split into train & val splits.
    2. train_source2.tsv and train_source3.tsv ARE NOT DOWNSAMPLED or filtered.
       The full S2 and S3 pools serve as the distractor search universe for validation.
    3. Stratification is based on country + match cardinality bucket (0-match, 1-match, 2+ match).
    4. Leakage checks verify set intersection of S1 IDs between train and val is empty.
    """
    train_dir = os.path.join(data_dir, "train")

    print(f"Loading training data from {train_dir}...")
    df_s1 = load_source_file(os.path.join(train_dir, "train_source1.tsv"))
    df_gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    # Merge S1 records with Ground Truth to obtain cardinality and country
    df_merged = pd.merge(df_s1, df_gt, left_on="entity_id", right_on="source1_entity_id", how="inner")

    # Compute match cardinality per S1 entity
    def get_cardinality(match_str: str) -> int:
        if not match_str or not match_str.strip():
            return 0
        return len([x for x in match_str.split(",") if x.strip()])

    df_merged["match_count"] = df_merged["matched_entity_ids"].apply(get_cardinality)

    # Create stratification key: country + cardinality bucket
    def get_strat_key(row):
        c = row["country"]
        cnt = row["match_count"]
        if cnt == 0:
            b = "0"
        elif cnt == 1:
            b = "1"
        else:
            b = "2plus"
        return f"{c}_{b}"

    df_merged["strat_key"] = df_merged.apply(get_strat_key, axis=1)

    print("\nStratification key distribution:")
    print(df_merged["strat_key"].value_counts())

    # Perform stratified 80/20 train/val split on S1 entities
    train_s1_merged, val_s1_merged = train_test_split(
        df_merged,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=df_merged["strat_key"]
    )

    print(f"\nSplit Completed:")
    print(f"  Train S1 Entities: {len(train_s1_merged):,}")
    print(f"  Val S1 Entities:   {len(val_s1_merged):,}")

    # Leakage Check 1: S1 entity ID set intersection
    train_s1_ids = set(train_s1_merged["entity_id"])
    val_s1_ids = set(val_s1_merged["entity_id"])
    s1_intersection = train_s1_ids.intersection(val_s1_ids)

    print(f"\nLeakage Verification:")
    print(f"  S1 ID Intersection (Train & Val): {len(s1_intersection)}")
    assert len(s1_intersection) == 0, "CRITICAL ERROR: S1 ID leakage detected between Train and Val!"

    # Leakage Check 2: Target S2/S3 entity IDs in ground truth
    def get_all_matched_targets(df_subset):
        targets = set()
        for m_str in df_subset["matched_entity_ids"]:
            if m_str and m_str.strip():
                for m_id in m_str.split(","):
                    if m_id.strip():
                        targets.add(m_id.strip())
        return targets

    train_target_ids = get_all_matched_targets(train_s1_merged)
    val_target_ids = get_all_matched_targets(val_s1_merged)
    target_intersection = train_target_ids.intersection(val_target_ids)

    print(f"  Unique S2/S3 ground truth targets in Train: {len(train_target_ids):,}")
    print(f"  Unique S2/S3 ground truth targets in Val:   {len(val_target_ids):,}")
    print(f"  Overlap of S2/S3 target IDs between Train and Val: {len(target_intersection):,}")

    # Prepare final clean split DataFrames
    s1_cols = ["entity_id", "business_name", "business_address", "country"]
    gt_cols = ["source1_entity_id", "matched_entity_ids"]

    train_s1_out = train_s1_merged[s1_cols].copy()
    train_gt_out = train_s1_merged[gt_cols].copy()

    val_s1_out = val_s1_merged[s1_cols].copy()
    val_gt_out = val_s1_merged[gt_cols].copy()

    # Save validation split files to output_dir
    os.makedirs(output_dir, exist_ok=True)
    val_split_train_dir = os.path.join(output_dir, "train_split")
    val_split_val_dir = os.path.join(output_dir, "val_split")

    os.makedirs(val_split_train_dir, exist_ok=True)
    os.makedirs(val_split_val_dir, exist_ok=True)

    print(f"\nSaving split files to {output_dir}...")
    train_s1_out.to_csv(os.path.join(val_split_train_dir, "train_source1.tsv"), sep="\t", index=False)
    train_gt_out.to_csv(os.path.join(val_split_train_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    val_s1_out.to_csv(os.path.join(val_split_val_dir, "val_source1.tsv"), sep="\t", index=False)
    val_gt_out.to_csv(os.path.join(val_split_val_dir, "val_ground_truth.tsv"), sep="\t", index=False)

    print("VALIDATION SPLIT CREATED & VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    default_data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "6ab10eb3b23ba_student_resource", "student_resource", "dataset"
    )
    default_out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "dataset", "split"
    )

    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_out_dir

    create_validation_split(os.path.abspath(data_dir), os.path.abspath(out_dir))
