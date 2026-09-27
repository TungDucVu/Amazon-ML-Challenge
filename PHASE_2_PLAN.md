# PHASE 2 IMPLEMENTATION PLAN: BUDGET-CONSTRAINED & HIGH-RECALL BLOCKING

## 1. Objective & Overview
Build a high-recall candidate generation pipeline that dramatically reduces the search space while adhering strictly to the **candidate pool budget penalty rule** (target average **5–15 candidates per S1 entity**, hard ceiling 20–25 candidates).

---

## 2. Detailed Task Breakdown

### Task 2.1: Country Hard-Blocking Partition Module (`blocking_country.py`)
- Restrict candidate pairs strictly within identical country labels:
  \[ Candidates(S1_i) \subseteq \{ S2, S3 \mid \text{country} = \text{country}(S1_i) \} \]
- Reduces candidate retrieval search space by 66–75% with **zero recall loss**.
- Handles `US` and `India` in training/validation, and `US`, `India`, `France` in test.

### Task 2.2: Multilingual Dense Vector Indexing (HNSW / FAISS)
- Model selection: Apache/MIT licensed multilingual embedding model (`BAAI/bge-m3` or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- Embed text strings formatted as: `"Country: {country} | Name: {business_name} | Address: {business_address}"`.
- Construct FAISS HNSW indexes per country partition for fast cosine similarity search.

### Task 2.3: Sparse TF-IDF N-Gram Cosine Indexing
- Extract character 3-grams and word 1-2 grams from `business_name` and `business_address`.
- Compute sparse TF-IDF vectors and build Cosine Top-K retrieval matrix using `scipy.sparse`.

### Task 2.4: Generic Postal & Phonetic Token Indexing
- Extract generic 4–6 digit isolated numbers (postal/PIN codes).
- Generate Double Metaphone / Soundex keys for primary business name tokens.

### Task 2.5: Dynamic Adaptive Filtering & Pool Budget Enforcer
- Combine candidate pairs from dense embeddings, TF-IDF n-grams, and postal/phonetic keys.
- Apply dynamic similarity thresholding:
  \[ \text{Include candidate if } \text{similarity} > 0.45 \text{ OR } \text{rank} \le 5 \]
- Enforce strict candidate pool budget: Cap at max 20–25 candidates per S1 entity, achieving an overall dataset average of **5–15 candidates per S1 entity**.

### Task 2.6: `candidate_pairs.tsv` Generator (`export_candidates.py`)
- Export final candidate pairs to `output/candidate_pairs.tsv`.
- Format: `source1_entity_id\tcandidate_entity_ids` (comma-separated S2-/S3- IDs or empty string).

---

## 3. Deliverables & Outputs
- `code/business_entity_resolution/src/blocking.py`
- `code/business_entity_resolution/src/export_candidates.py`
- Generated candidate file `output/candidate_pairs.tsv` for validation and test sets.

---

## 4. Manual Quality Control (QC) Gate & Stop Instruction

> [!IMPORTANT]
> **STOP & PERFORM MANUAL QUALITY CONTROL BEFORE PROCEEDING TO PHASE 3**

### Step-by-Step QC Execution Instructions:

1. **Run Candidate Generation Pipeline**:
   ```bash
   python code/business_entity_resolution/src/blocking.py --data-dir dataset/val_split --out output/val_candidate_pairs.tsv
   ```

2. **Audit Candidate Pool Size & Efficiency**:
   Execute the following Python audit snippet to check candidate pool statistics:
   ```python
   import pandas as pd
   df = pd.read_csv("output/val_candidate_pairs.tsv", sep="\t")
   df['count'] = df['candidate_entity_ids'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
   print(f"Average Candidates per S1: {df['count'].mean():.2f}")
   print(f"Max Candidates per S1: {df['count'].max()}")
   print(f"Percentage of S1 with >25 Candidates: {(df['count'] > 25).mean() * 100:.2f}%")
   ```
   *Manual Verification Checklist*:
   - [ ] **Average candidate pool size is strictly between 5.0 and 15.0**.
   - [ ] Max candidate pool size per S1 entity does not exceed 25.

3. **Verify Recall Ceiling on Validation Split**:
   ```bash
   python code/business_entity_resolution/src/blocking.py --evaluate-recall --ground-truth dataset/val_split/val_ground_truth.tsv --candidates output/val_candidate_pairs.tsv
   ```
   *Manual Verification Checklist*:
   - [ ] Confirm blocking recall ceiling is **> 98.0%** (i.e. over 98% of true ground-truth matches exist within the generated candidate pool).

4. **Verify Candidate Output Format**:
   - [ ] Confirm `output/val_candidate_pairs.tsv` uses tab separator (`\t`).
   - [ ] Confirm every S1 entity in validation set appears exactly once.

**Decision Checkpoint**: Once candidate pool size (5–15 average) and recall ceiling (>98%) are confirmed, inform the assistant to proceed with **Phase 3 Implementation**.
