import os
import pandas as pd
from collections import Counter

def load_and_analyze(path, name):
    print(f"\n--- {name} ---")
    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}")
        return None
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    print(f"Row count: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if 'country' in df.columns:
        print(f"Country distribution:\n{df['country'].value_counts().to_string()}")
        
    for col in df.columns:
        missing = (df[col] == "").sum()
        if missing > 0:
            print(f"Missing {col}: {missing} ({missing/len(df):.2%})")
            
    if 'entity_id' in df.columns:
        dups = df['entity_id'].duplicated().sum()
        if dups > 0:
            print(f"Duplicate entity_ids: {dups}")
    
    return df

def analyze_ground_truth(gt_df):
    print("\n--- Ground Truth Analysis ---")
    if gt_df is None: return
    
    match_counts = gt_df['matched_entity_ids'].apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x.strip() else 0)
    print("Match Cardinality Distribution:")
    print(match_counts.value_counts().sort_index().to_string())
    
    # S2 vs S3 matches
    s2_matches = 0
    s3_matches = 0
    for ids in gt_df['matched_entity_ids']:
        if ids.strip():
            id_list = [i.strip() for i in ids.split(',')]
            s2_matches += sum(1 for i in id_list if i.startswith('S2-'))
            s3_matches += sum(1 for i in id_list if i.startswith('S3-'))
    print(f"Total S2 Matches: {s2_matches}")
    print(f"Total S3 Matches: {s3_matches}")

if __name__ == '__main__':
    train_s1 = load_and_analyze('dataset/train/train_source1.tsv', 'Train S1')
    train_s2 = load_and_analyze('dataset/train/train_source2.tsv', 'Train S2')
    train_s3 = load_and_analyze('dataset/train/train_source3.tsv', 'Train S3')
    gt = load_and_analyze('dataset/train/train_ground_truth.tsv', 'Train Ground Truth')
    
    test_s1 = load_and_analyze('dataset/test/test_source1.tsv', 'Test S1')
    test_s2 = load_and_analyze('dataset/test/test_source2.tsv', 'Test S2')
    test_s3 = load_and_analyze('dataset/test/test_source3.tsv', 'Test S3')
    
    analyze_ground_truth(gt)
