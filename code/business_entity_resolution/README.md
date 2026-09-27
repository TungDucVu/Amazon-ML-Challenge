# Amazon ML Challenge 2026: Business Entity Resolution

## Quick Start

### Prerequisites
- Python 3.10+ 
- Virtual environment recommended

### Setup
```bash
cd "amazon ml"
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

### Run Full Pipeline
```bash
# With your own dataset (place files in dataset/train and dataset/test)
python run_pipeline.py --data-dir dataset --output-dir output

# With synthetic sample data (for testing the pipeline)
python run_pipeline.py --data-dir dataset --output-dir output --generate-sample
```

### Validate Submission
```bash
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline Architecture

```
Phase 1: Data Loading & Validation
    ├── Strict TSV parsing (sep="\t")
    ├── Schema validation (entity_id, business_name, business_address, country)
    ├── Open-set country partitioning (US, India, France)
    └── Stratified 80/20 train/val split (by country + cardinality)

Phase 2: Candidate Generation (Blocking)
    ├── Exact Country Hard-Blocking Partition (66-75% space reduction)
    ├── Character 3-4 gram TF-IDF cosine retrieval
    ├── Word 1-2 gram TF-IDF cosine retrieval
    ├── Generic numeric/postal token exact-match index
    └── Adaptive Dynamic Top-K filtering (budget: 5-15 candidates/S1)

Phase 3: Feature Engineering (24 features)
    ├── Name: Levenshtein, Jaro-Winkler, Token Sort/Set Ratio, n-gram Jaccard/Dice
    ├── Address: Levenshtein, Token Sort/Set Ratio, Token Jaccard, n-gram Jaccard
    ├── Numeric: Postal/PIN Jaccard, shared count, digit string similarity
    ├── Legal: Suffix match status, suffix Jaccard
    └── Combined: Full-field Token Sort/Partial Ratio, length ratio, token count diff

Phase 4: ML Matching Classifier
    ├── LightGBM + XGBoost ensemble (0.5/0.5 weighting)
    ├── Class imbalance handling (scale_pos_weight)
    └── Precision-heavy threshold optimization for Macro F_0.5

Phase 5: Post-Processing & Output
    ├── Singleton safeguard (empty match for low-confidence entities)
    ├── Subset constraint enforcement (matching ⊆ candidate)
    ├── Output format verification
    └── Official validate_submission.py PASS check
```

## Source Code Structure

```
code/business_entity_resolution/
├── requirements.txt          # Pinned dependencies
├── README.md                 # This file
└── src/
    ├── __init__.py
    ├── data_loader.py        # TSV loading & schema validation
    ├── preprocessing.py      # Text normalization & tokenization
    ├── blocking.py           # Multi-stage candidate generation
    ├── features.py           # Pairwise feature engineering
    ├── matcher.py            # GBDT model training & threshold optimization
    ├── metrics.py            # Macro F_0.5 evaluation harness
    ├── split.py              # Stratified train/val split
    ├── synthetic_data.py     # Benchmark data generator
    └── postprocessing.py     # Output formatting & validation
```

## Output Files

| File | Description |
|---|---|
| `output/matching_results.tsv` | Final entity matches (scored on leaderboard) |
| `output/candidate_pairs.tsv` | Blocking candidate set (audited for efficiency) |

## Compliance

- ✅ Zero external API calls or data lookups
- ✅ All models < 8B parameters (LightGBM + XGBoost only)
- ✅ MIT/Apache 2.0 license compliance
- ✅ Tab-separated parsing enforced
- ✅ Open-set: France processed seamlessly
- ✅ Official validate_submission.py: PASS
