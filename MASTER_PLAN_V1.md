# MASTER IMPLEMENTATION PLAN V1: AMAZON ML CHALLENGE 2026 - BUSINESS ENTITY RESOLUTION

## 1. Executive Summary & Architecture Overview

This master plan details the end-to-end architecture, methodology, implementation steps, and validation strategy for building a high-precision, open-set **Business Entity Resolution (ER)** system.

The objective is to match records from Source 2 (`S2-`) and Source 3 (`S3-`) to deduplicated reference entities in Source 1 (`S1-`). The pipeline must handle severe data noise (abbreviations, typos, missing address fields, landmark references), generalize to unseen countries (`France` in test set), operate under strict rules (zero external data lookup, <8B parameters, Apache 2.0/MIT licenses), and optimize specifically for the precision-heavy **Macro-averaged \(F_{0.5}\)** metric.

---

## 2. Pipeline & System Architecture

```mermaid
flowchart TD
    subgraph Data Ingestion & Preprocessing
        A[Raw TSV Files S1, S2, S3] --> B[Text Normalizer & Cleaner]
        B --> C[Country-Aware & Standardized Schema]
    end

    subgraph Candidate Generation (Blocking)
        C --> D1[Sparse TF-IDF N-Gram Indexing]
        C --> D2[Dense Embedding HNSW Indexing]
        C --> D3[Phonetic & Postal Code Indexing]
        D1 & D2 & D3 --> E[Candidate Union & Filtering]
        E --> F[candidate_pairs.tsv Generator]
    end

    subgraph Feature Engineering
        F --> G1[String Similarity Features: Levenshtein, Jaro-Winkler, Jaccard]
        F --> G2[Semantic Vector Cosine Distance]
        F --> G3[Address & Digits/PIN Overlap Metrics]
        F --> G4[Phonetic & Legal Suffix Overlap]
    end

    subgraph Matching Model & Post-Processing
        G1 & G2 & G3 & G4 --> H[GBDT Ensemble: LightGBM / XGBoost / CatBoost]
        H --> I[Precision-Calibrated Thresholding for F_0.5]
        I --> J[Singleton Filter & Format Enforcer]
    end

    subgraph Submission Packaging
        J --> K1[output/matching_results.tsv]
        F --> K2[output/candidate_pairs.tsv]
        K1 & K2 --> L[utils/validate_submission.py Verification]
        L --> M[Zip Submission Package Generation]
    end
```

---

## 3. Phase-by-Phase Implementation Plan

### Phase 1: Infrastructure, EDA & Validation Setup
- **Directory & Repo Setup**:
  - Link repository with GitHub remote `https://github.com/TungDucVu/Amazon-ML-Challenge.git`.
  - Maintain clean folder structure under `code/business_entity_resolution/src/`.
- **Data Loaders & Tab-Separator Safeguards**:
  - Implement robust `.tsv` reader ensuring explicit `sep="\t"` parsing.
  - Implement schema validation script checking column names (`entity_id`, `business_name`, `business_address`, `country`).
- **Exploratory Data Analysis (EDA)**:
  - Calculate entity distributions across sources (`S1`, `S2`, `S3`).
  - Analyze train ground truth match cardinality (ratio of 0-match singletons vs 1-match vs multi-match entities).
  - Inspect noise types: legal suffix frequency (`Inc`, `LLC`, `Pvt Ltd`), common typos, address format variations between `US` and `India`.
- **Validation Split Strategy**:
  - Create a representative local validation split from `train_ground_truth.tsv` (e.g., 80% train / 20% validation) stratified by country and match cardinality.
  - Implement local Macro \(F_{0.5}\) calculation harness identical to competition metric.

### Phase 2: High-Recall Multi-Stage Candidate Generation (Blocking)
- **Goal**: Achieve **>98% recall ceiling** on candidate pairs while keeping candidate set manageable (<100 candidates per S1 entity).
- **Stage 2.1: Rule-Based & Phonetic Blocking**:
  - Exact/Fuzzy PIN/Postal code match per country partition.
  - Double Metaphone / Soundex phonetic key blocking on business name primary tokens.
- **Stage 2.2: Sparse TF-IDF Character & Word N-Gram Indexing**:
  - Character 3-gram and word 1-2 gram TF-IDF vectorization for names and addresses.
  - Top-K cosine similarity retrieval (e.g., K=30 per source) using sparse matrix multiplication (`scipy.sparse`).
- **Stage 2.3: Dense Embedding Vector Indexing (FAISS / HNSW)**:
  - Generate dense embeddings for concatenated `[country, business_name, business_address]` using an open-source Transformer model (<8B params, MIT/Apache 2.0 license, e.g., `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-base-en-v1.5`).
  - Build FAISS flat/HNSW index for fast top-K nearest neighbor search.
- **Stage 2.4: Candidate Union & Output Generation**:
  - Take union of all blocking candidates per Source 1 entity.
  - Generate and validate `candidate_pairs.tsv`.

### Phase 3: Comprehensive Multi-Dimensional Feature Engineering
For each `(Source 1, Candidate)` pair in `candidate_pairs.tsv`, compute engineered feature vectors:
- **Name Similarity Features**:
  - Normalized Levenshtein distance, Damerau-Levenshtein distance.
  - Jaro-Winkler similarity.
  - Character n-gram Jaccard similarity & Dice coefficient.
  - Token sort ratio & Token set ratio (`thefuzz` / `RapidFuzz`).
  - Legal suffix matching indicator (match, mismatch, missing).
- **Address & Structural Features**:
  - Address token Jaccard & Soft-TFIDF overlap.
  - House/Building number exact digit match flag.
  - Postal/PIN code match status (exact match, prefix match, mismatch, missing).
  - Street/Road name string similarity.
- **Semantic & Phonetic Features**:
  - Dense embedding cosine similarity and dot product.
  - Phonetic distance between primary name tokens.
- **Open-Set Generalization Safeguards**:
  - All engineered features MUST be relative similarity ratios (0.0 to 1.0) or language-agnostic distance metrics.
  - Avoid country-specific hardcoded dictionary dependencies so feature distributions remain consistent for `France` in the test set.

### Phase 4: Machine Learning Matching Classifier & Threshold Optimization
- **Model Choice**: GBDT ensemble combining **LightGBM**, **XGBoost**, and **CatBoost**.
- **Training Strategy**:
  - Construct binary classification dataset where true ground truth pairs are `1` and non-matching candidate pairs are `0`.
  - Handle class imbalance via scale_pos_weight or focal loss objective.
- **Threshold Calibration for Macro \(F_{0.5}\)**:
  - Predict pairwise match probability \(P(\text{match})\).
  - Search optimal classification threshold \(\theta^*\) on the validation set specifically maximizing the Macro \(F_{0.5}\) score.
  - Since \(F_{0.5}\) weights precision 2x higher than recall, set conservative thresholds to minimize false positive merges.

### Phase 5: Post-Processing, Singleton Handling & Submission Verification
- **Singleton Detection**:
  - Entities where no candidate exceeds threshold \(\theta^*\) are assigned an empty match list (`""`).
  - Ensure singletons are preserved correctly to earn 1.0 scores on 0-match entities.
- **Format Verification**:
  - Format predictions into `output/matching_results.tsv`.
  - Enforce constraint: `matching_results.tsv` matches MUST be a strict subset of `candidate_pairs.tsv`.
  - Enforce constraint: No duplicate IDs per row, valid S2-/S3- entity prefixes only.
  - Run `utils/validate_submission.py` locally:
    ```bash
    python3 utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test
    ```

### Phase 6: Submission Packaging & Documentation
- **Code Organization**:
  - Place all runnable code inside `code/business_entity_resolution/src/`.
  - Write step-by-step reproduction instructions in `code/business_entity_resolution/README.md`.
  - Create pinned `code/business_entity_resolution/requirements.txt`.
- **Documentation**:
  - Complete `Documentation_template.md` covering problem analysis, blocking strategy, model features, and validation results.
- **Zip Archiving**:
  - Package final submission into `<team_name>_submission.zip`.

---

## 4. Key Milestones & Project Deliverables

| Milestone | Deliverable | Target Timeline | Status |
|---|---|---|---|
| **M0** | Requirements summary (`REQUIREMENT.txt`), Master Plan (`MASTER_PLAN_V1.md`), Git remote setup | Day 1 | Completed |
| **M1** | Data ingestion pipeline, local validation split & F_0.5 scoring script | Day 2 | Pending |
| **M2** | Multi-stage blocking module & baseline `candidate_pairs.tsv` (>95% recall) | Day 3 | Pending |
| **M3** | Feature engineering pipeline (string, address, embedding similarity) | Day 4 | Pending |
| **M4** | GBDT model training, hyperparameter tuning & threshold calibration | Day 5 | Pending |
| **M5** | Validation harness execution, `utils/validate_submission.py` PASS verification | Day 6 | Pending |
| **M6** | Submission package zip creation (`code/`, `output/`, `Documentation_template.md`) | Day 7 | Pending |

---

## 5. Compliance & Quality Assurance Checklist

- [x] Repository initialized & linked with remote `https://github.com/TungDucVu/Amazon-ML-Challenge.git`.
- [ ] Strictly zero external API calls or external database lookups during training or inference.
- [ ] Open-source model license compliance (MIT / Apache 2.0) and size under 8 Billion parameters.
- [ ] Tab-separated parsing (`sep="\t"`) verified for all data readers and TSV generators.
- [ ] Open-set test handling verified (pipeline processes `France` records seamlessly).
- [ ] Submission files pass `utils/validate_submission.py` with zero errors.
