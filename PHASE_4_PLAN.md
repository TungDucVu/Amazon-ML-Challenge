# PHASE 4 IMPLEMENTATION PLAN: ML MATCHING CLASSIFIER & THRESHOLD OPTIMIZATION

## 1. Objective & Overview
Train a high-performance Gradient Boosted Decision Tree (GBDT) ensemble model, tune classification decision threshold \(\theta^*\) to optimize **Macro \(F_{0.5}\)** (2x precision-weighted), and enforce source-level uniqueness constraints to protect singleton 1.0 scores.

---

## 2. Detailed Task Breakdown

### Task 4.1: GBDT Ensemble Training Module (`train.py`)
- Location: `code/business_entity_resolution/src/train.py`
- Train ensemble models: **LightGBM**, **XGBoost**, and **CatBoost**.
- Hyperparameter tuning:
  - Objective: Binary classification (`binary:logistic` / `binary_logloss`).
  - Imbalance handling: Adjust `scale_pos_weight` / `pos_bagging_fraction` since candidate sets contain ~90% negative pairs.
  - Cross-Validation: 5-fold cross-validation on training candidate features.

### Task 4.2: Macro \(F_{0.5}\) Threshold Grid Search (`tune_threshold.py`)
- Evaluate decision threshold \(\theta^* \in [0.50, 0.90]\) step 0.01 on the validation set.
- Target: Find \(\theta^*\) that maximizes full Macro \(F_{0.5}\) score across all S1 entities (including singletons).
- Expect \(\theta^*\) to skew high (\(\approx 0.65 - 0.80\)) due to \(F_{0.5}\) weighting precision 2x over recall.

### Task 4.3: Relative Margin & Source Uniqueness Constraint Module (`post_filter.py`)
- **Source Uniqueness Limit**: In real-world business ER, an S1 entity rarely matches multiple records from the same source file.
- Rule: Enforce at most 1 match per source file (`S2-`, `S3-`) for each S1 entity unless top candidate probabilities are within a tiny margin (\(\Delta < 0.03\)).

### Task 4.4: Singleton Protection Mechanism
- If top predicted match probability for an S1 entity is below \(\theta^*\), predict empty match list `""`.
- Ensures singletons with 0 true matches achieve maximum 1.0 credit instead of being penalized 0.0 by false positives.

### Task 4.5: Model Checkpointing & Inference Pipeline (`predict.py`)
- Save trained GBDT models to `models/gbdt_ensemble.pkl`.
- Implement standalone inference module `predict.py` to generate matching probabilities for test candidate pairs.

---

## 3. Deliverables & Outputs
- `code/business_entity_resolution/src/train.py`
- `code/business_entity_resolution/src/tune_threshold.py`
- `code/business_entity_resolution/src/predict.py`
- Saved model file `models/gbdt_ensemble.pkl`

---

## 4. Manual Quality Control (QC) Gate & Stop Instruction

> [!IMPORTANT]
> **STOP & PERFORM MANUAL QUALITY CONTROL BEFORE PROCEEDING TO PHASE 5**

### Step-by-Step QC Execution Instructions:

1. **Run Model Training & Hyperparameter Tuning**:
   ```bash
   python code/business_entity_resolution/src/train.py --features features_train.parquet --val-features features_val.parquet --model-dir models/
   ```

2. **Execute Threshold Search & Macro \(F_{0.5}\) Evaluation**:
   ```bash
   python code/business_entity_resolution/src/tune_threshold.py --model models/gbdt_ensemble.pkl --val-features features_val.parquet --val-ground-truth dataset/val_split/val_ground_truth.tsv
   ```
   *Manual Verification Checklist*:
   - [ ] Confirm optimal decision threshold \(\theta^*\) skews high (\(\theta^* \ge 0.65\)).
   - [ ] Verify validation **Macro \(F_{0.5}\) score is printed and logged** (target > 0.85).
   - [ ] Inspect feature importance plot/output (verify top features include name string similarity, multilingual vector cosine, and postal digit overlap).

3. **Audit Singleton Prediction Accuracy**:
   Run python inspection command:
   ```python
   import pandas as pd
   # Load validation predictions and ground truth
   val_gt = pd.read_csv("dataset/val_split/val_ground_truth.tsv", sep="\t")
   val_pred = pd.read_csv("output/val_matching_results.tsv", sep="\t")
   merged = pd.merge(val_gt, val_pred, on="source1_entity_id", suffixes=("_gt", "_pred"))
   singletons = merged[merged['matched_entity_ids_gt'].isna() | (merged['matched_entity_ids_gt'] == '')]
   correct_singletons = (singletons['matched_entity_ids_pred'].isna() | (singletons['matched_entity_ids_pred'] == '')).mean() * 100
   print(f"Singleton Accuracy (Target >95%): {correct_singletons:.2f}%")
   ```
   *Manual Verification Checklist*:
   - [ ] **Singleton prediction accuracy is >95%** (correctly predicting empty matches for true singletons).

**Decision Checkpoint**: Once Macro \(F_{0.5}\) score and >95% singleton accuracy are verified, inform the assistant to proceed with **Phase 5 Implementation**.
