import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from data_loader import load_ground_truth, load_source_file

# Paths to generated split files
train_s1_path = os.path.abspath("dataset/split/train_split/train_source1.tsv")
val_s1_path = os.path.abspath("dataset/split/val_split/val_source1.tsv")
train_gt_path = os.path.abspath("dataset/split/train_split/train_ground_truth.tsv")
val_gt_path = os.path.abspath("dataset/split/val_split/val_ground_truth.tsv")

print("Loading Train and Val Ground Truth files...")
df_train_gt = load_ground_truth(train_gt_path)
df_val_gt = load_ground_truth(val_gt_path)

df_train_s1 = load_source_file(train_s1_path)
df_val_s1 = load_source_file(val_s1_path)

# Function to parse IDs by prefix
def extract_target_ids(df_gt):
    s2_set = set()
    s3_set = set()
    for m_str in df_gt["matched_entity_ids"]:
        if m_str and m_str.strip():
            for m_id in m_str.split(","):
                m_id = m_id.strip()
                if m_id.startswith("S2-"):
                    s2_set.add(m_id)
                elif m_id.startswith("S3-"):
                    s3_set.add(m_id)
    return s2_set, s3_set

train_s2_targets, train_s3_targets = extract_target_ids(df_train_gt)
val_s2_targets, val_s3_targets = extract_target_ids(df_val_gt)

s2_shared = train_s2_targets.intersection(val_s2_targets)
s3_shared = train_s3_targets.intersection(val_s3_targets)
total_shared = (train_s2_targets.union(train_s3_targets)).intersection(val_s2_targets.union(val_s3_targets))

print("\n================================================================================")
print("GROUND-TRUTH TARGET ID OVERLAP AUDIT (S2 / S3 TARGET LEAKAGE AUDIT)")
print("================================================================================\n")
print(f"Source 2 Ground-Truth Target IDs:")
print(f"  Train S2 Targets: {len(train_s2_targets):,}")
print(f"  Val S2 Targets:   {len(val_s2_targets):,}")
print(f"  Shared S2 Targets (Train AND Val): {len(s2_shared):,} ({len(s2_shared)/len(val_s2_targets)*100:.4f}% of Val S2 targets, {len(s2_shared)/len(train_s2_targets)*100:.4f}% of Train S2 targets)")

print(f"\nSource 3 Ground-Truth Target IDs:")
print(f"  Train S3 Targets: {len(train_s3_targets):,}")
print(f"  Val S3 Targets:   {len(val_s3_targets):,}")
print(f"  Shared S3 Targets (Train AND Val): {len(s3_shared):,} ({len(s3_shared)/len(val_s3_targets)*100:.4f}% of Val S3 targets, {len(s3_shared)/len(train_s3_targets)*100:.4f}% of Train S3 targets)")

print(f"\nCombined S2 + S3 Ground-Truth Target IDs:")
print(f"  Train Combined Targets: {len(train_s2_targets) + len(train_s3_targets):,}")
print(f"  Val Combined Targets:   {len(val_s2_targets) + len(val_s3_targets):,}")
print(f"  Shared Combined Targets (Train AND Val): {len(total_shared):,} ({len(total_shared)/(len(val_s2_targets)+len(val_s3_targets))*100:.4f}% of Val combined targets)")


# Stratification breakdown calculation
print("\n================================================================================")
print("STRATIFICATION BUCKET PERCENTAGE BREAKDOWN (TRAIN VS VAL)")
print("================================================================================\n")

df_train_merged = pd.merge(df_train_s1, df_train_gt, left_on="entity_id", right_on="source1_entity_id")
df_val_merged = pd.merge(df_val_s1, df_val_gt, left_on="entity_id", right_on="source1_entity_id")

def assign_strat_bucket(df):
    def get_cnt(match_str):
        if not match_str or not match_str.strip():
            return 0
        return len([x for x in match_str.split(",") if x.strip()])

    df["match_count"] = df["matched_entity_ids"].apply(get_cnt)

    def bucket_fn(row):
        c = row["country"]
        cnt = row["match_count"]
        b = "0" if cnt == 0 else ("1" if cnt == 1 else "2plus")
        return f"{c}_{b}"

    df["strat_bucket"] = df.apply(bucket_fn, axis=1)

assign_strat_bucket(df_train_merged)
assign_strat_bucket(df_val_merged)

train_counts = df_train_merged["strat_bucket"].value_counts()
train_pcts = (train_counts / len(df_train_merged) * 100).round(4)

val_counts = df_val_merged["strat_bucket"].value_counts()
val_pcts = (val_counts / len(df_val_merged) * 100).round(4)

summary_df = pd.DataFrame({
    "Train Count": train_counts,
    "Train %": train_pcts,
    "Val Count": val_counts,
    "Val %": val_pcts
}).sort_index()

print(summary_df.to_string())
