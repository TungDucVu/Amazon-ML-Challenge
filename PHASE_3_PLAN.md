# PHASE 3 IMPLEMENTATION PLAN: LANGUAGE-AGNOSTIC FEATURE ENGINEERING

## 1. Objective & Overview
Engineer a rich, multi-dimensional feature matrix for every candidate pair `(Source 1, Candidate)` in `candidate_pairs.tsv`. All features MUST be **language-agnostic** to guarantee zero performance drop when transferring to unseen countries like **`France`**.

---

## 2. Detailed Task Breakdown

### Task 3.1: Language-Agnostic String Similarity Extractor (`feature_string.py`)
- Compute pair similarity metrics on `business_name`:
  - Normalized Levenshtein distance & Damerau-Levenshtein distance.
  - Jaro-Winkler similarity score.
  - Character 3-gram and 4-gram Dice coefficient & Jaccard overlap.
  - RapidFuzz token sort ratio and token set ratio.
  - Legal suffix matching indicator (match, mismatch, missing).

### Task 3.2: Numeric & Address Structural Feature Extractor (`feature_address.py`)
- Extract and compare address elements without language-specific regexes:
  - **Digit Sequence Match %**: Extract all digits from S1 and Candidate address strings; calculate exact digit sequence similarity ratio.
  - **Generic Postal Token Match**: Extract any 4–6 digit isolated numbers; return binary match flag (1 if match, 0 otherwise).
  - **Street Token Overlap**: Jaccard token overlap for address strings (handles *Rue*, *Boulevard*, *Avenue*, *Road*, *Street* identically).
  - Address token length ratio & character length ratio.

### Task 3.3: Multilingual Vector Embedding Similarity Extractor (`feature_embeddings.py`)
- Embed full record representations using Apache/MIT licensed multilingual transformer model (`BAAI/bge-m3` or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- Features generated:
  - Cosine similarity between S1 embedding and Candidate embedding.
  - Dot product & L2 distance between S1 embedding and Candidate embedding.

### Task 3.4: Phonetic & Token Overlap Extractor (`feature_phonetic.py`)
- Double Metaphone / Soundex key overlap ratio on business name tokens.

### Task 3.5: Pairwise Feature Matrix Builder & Parquet Caching (`build_feature_matrix.py`)
- Combine all features into tabular DataFrame.
- Store features in efficient Parquet format: `features_train.parquet`, `features_val.parquet`, `features_test.parquet`.

---

## 3. Deliverables & Outputs
- `code/business_entity_resolution/src/features.py`
- Cached feature files: `features_val.parquet`, `features_test.parquet`.

---

## 4. Manual Quality Control (QC) Gate & Stop Instruction

> [!IMPORTANT]
> **STOP & PERFORM MANUAL QUALITY CONTROL BEFORE PROCEEDING TO PHASE 4**

### Step-by-Step QC Execution Instructions:

1. **Run Feature Extraction Pipeline**:
   ```bash
   python code/business_entity_resolution/src/build_feature_matrix.py --candidates output/val_candidate_pairs.tsv --data-dir dataset/val_split --out features_val.parquet
   ```

2. **Audit Feature Data Quality**:
   Run the following Python script to check feature integrity:
   ```python
   import pandas as pd
   df = pd.read_parquet("features_val.parquet")
   print("Feature Matrix Shape:", df.shape)
   print("\nMissing Values per Feature:")
   print(df.isnull().sum()[df.isnull().sum() > 0])
   print("\nSample Feature Statistics:")
   print(df.describe().T[['min', 'mean', 'max']])
   ```
   *Manual Verification Checklist*:
   - [ ] Confirm total row count matches number of candidate pairs generated in Phase 2.
   - [ ] Confirm **zero NaN, Null, or Infinite (`inf`) values** exist in feature columns (fill missing numeric values with 0.0).
   - [ ] Confirm all string similarity features fall in range `[0.0, 1.0]`.

3. **Verify Language-Agnostic Design**:
   - [ ] Confirm no hardcoded US state lists or Indian PIN code lookups were used in feature code.
   - [ ] Verify digit sequence match % handles arbitrary address formats (including French postal code structures).

**Decision Checkpoint**: Once feature matrix integrity (0 NaNs, correct shape) is verified, inform the assistant to proceed with **Phase 4 Implementation**.
