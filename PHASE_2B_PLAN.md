# PHASE 2B PLAN: RECALL CEILING & PARETO CURVE OPTIMIZATION

## 1. Objective & Directives

Phase 2 Baseline v1 established a high-throughput engineering pipeline (13.6 mins for 441K queries vs 10.3M targets), but achieved **83.86% Target-Level Recall**, leaving 16.14% of ground-truth relationships unrecoverable. 

**Phase 2B** executes 4 core directives to push Target-Level Recall to **$\ge 95\% - 98\%$** before advancing to Phase 3 feature engineering:

1. **Dense Vector Channel (`paraphrase-multilingual-MiniLM-L12-v2` + FAISS)**:
   - Generate 384-dimensional dense vectors for entities formatted as `"Country: {country} | Name: {business_name} | Address: {business_address}"`.
   - Build country-partitioned FAISS `IndexFlatIP` indices to retrieve Top 15 dense nearest neighbors.

2. **Widen Pre-Union Top-K Quotas Across Sparse Channels**:
   - Expand `sparse_dot_topn` channel quotas:
     - **Channel 1 (Word Name+Addr)**: Top 20
     - **Channel 2 (Word Name Only)**: Top 15
     - **Channel 3 (Char_wb Name Only)**: Top 15
   - Form multi-modal candidate union: $\text{Candidate Pool} = \text{Dense} \cup \text{Sparse}_1 \cup \text{Sparse}_2 \cup \text{Sparse}_3$.

3. **Trace Empirical Pareto Frontier Across $K \in [10, 15, 20, 25, 30, 40]$**:
   - Measure Target-Level Recall, Complete-S1 Recall, and Average Candidate Pool Size across post-union deduplicated caps $K \in [10, 15, 20, 25, 30, 40]$.
   - Locate the optimal Pareto "elbow" operating point where Target Recall reaches $\ge 95\% - 98\%$.

4. **Open-Set Generalization Dry Run (`France`)**:
   - Execute country partition logic on synthetic `France` records to verify zero hardcoding or country assumption leakage.

---

## 2. Implementation Steps & Deliverables

- **Code Update**: Update `code/business_entity_resolution/src/blocking.py` to support multi-modal dense + sparse retrieval and Pareto evaluation harness.
- **Pareto Audit Script**: `scratch/evaluate_pareto_curve.py` to trace recall vs $K$.
- **Documentation**: Update `docs/PHASE_2_REPORT.md` with Phase 2B results and Pareto curve metrics.
- **Deliverable**: High-recall candidate file `output/val_candidate_pairs.tsv` satisfying $\ge 95\%$ Target Recall.
