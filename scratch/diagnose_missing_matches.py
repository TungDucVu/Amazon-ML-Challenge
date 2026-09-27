import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from data_loader import load_source_file, load_ground_truth

if __name__ == "__main__":
    val_s1_path = os.path.abspath("dataset/split/val_split/val_source1.tsv")
    val_gt_path = os.path.abspath("dataset/split/val_split/val_ground_truth.tsv")
    s2_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv")
    s3_path = os.path.abspath("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv")

    print("Loading data...", flush=True)
    df_s1 = load_source_file(val_s1_path)
    df_gt = load_ground_truth(val_gt_path)
    df_cand = pd.read_csv(os.path.abspath("output/val_candidate_pairs.tsv"), sep="\t", dtype=str, keep_default_na=False)

    cand_map = dict(zip(df_cand["source1_entity_id"], df_cand["candidate_entity_ids"]))
    s1_map = df_s1.set_index("entity_id").to_dict("index")

    # Find first 15 missed pairs
    missed_pairs = []
    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        if not m_str or not m_str.strip():
            continue
        true_targets = [x.strip() for x in m_str.split(",") if x.strip()]
        cands = set([x.strip() for x in cand_map.get(s1_id, "").split(",") if x.strip()])

        for t_id in true_targets:
            if t_id not in cands:
                missed_pairs.append((s1_id, t_id))
                if len(missed_pairs) >= 15:
                    break
        if len(missed_pairs) >= 15:
            break

    print(f"Identified {len(missed_pairs)} missed pairs to inspect.", flush=True)
    missed_target_ids = set([t_id for _, t_id in missed_pairs])

    print("Filtering target sources...", flush=True)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)
    df_targets = pd.concat([df_s2[df_s2["entity_id"].isin(missed_target_ids)], df_s3[df_s3["entity_id"].isin(missed_target_ids)]], ignore_index=True)
    target_map = df_targets.set_index("entity_id").to_dict("index")

    print("\n=== MISSED GROUND TRUTH PAIRS DIAGNOSIS ===", flush=True)
    for idx, (s1_id, t_id) in enumerate(missed_pairs, 1):
        s1_rec = s1_map.get(s1_id, {})
        t_rec = target_map.get(t_id, {})
        print(f"\n[Missed Pair #{idx}] S1: {s1_id} <--> Target: {t_id}", flush=True)
        print(f"  S1 Name:     '{s1_rec.get('business_name')}'", flush=True)
        print(f"  S1 Address:  '{s1_rec.get('business_address')}'", flush=True)
        print(f"  Target Name: '{t_rec.get('business_name')}'", flush=True)
        print(f"  Target Addr: '{t_rec.get('business_address')}'", flush=True)
