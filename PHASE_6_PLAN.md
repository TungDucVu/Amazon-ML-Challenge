# PHASE 6 IMPLEMENTATION PLAN: SUBMISSION PACKAGING & DELIVERABLES

## 1. Objective & Overview
Structure the self-contained, reproducible codebase under `code/business_entity_resolution/`, pin environment dependencies, fill out `Documentation_template.md`, and generate the final submission zip package `<team_name>_submission.zip`.

---

## 2. Detailed Task Breakdown

### Task 6.1: Runnable Codebase Structuring (`code/business_entity_resolution/`)
- Ensure all source code modules are organized cleanly:
  ```
  code/business_entity_resolution/
  ├── src/
  │   ├── data_loader.py
  │   ├── eda.py
  │   ├── blocking.py
  │   ├── features.py
  │   ├── train.py
  │   ├── predict.py
  │   ├── export_candidates.py
  │   ├── export_matching.py
  │   └── verify_subset.py
  ├── main.py                     # Single end-to-end master execution script
  ├── README.md                   # Step-by-step reproduction guide
  └── requirements.txt            # Pinned dependencies
  ```

### Task 6.2: End-to-End Master Runner & README (`main.py` & `README.md`)
- `main.py`: Accepts `--data-dir dataset/test` and automatically runs blocking -> feature extraction -> model inference -> post-processing -> validation.
- `README.md`: Explains how to set up environment and run end-to-end execution command.

### Task 6.3: Methodology Document Writing (`Documentation_template.md`)
- Complete all required sections in `Documentation_template.md`:
  1. Executive Summary
  2. Methodology & Problem Analysis
  3. Candidate Generation (Blocking Keys, candidate count reduction, recall ceiling)
  4. Matching Model (Feature engineering, GBDT ensemble, threshold selection)
  5. Results & Error Analysis
  6. Conclusion & Appendix (Code artifacts summary)

### Task 6.4: Submission Package Zip Generator (`package_submission.py`)
- Location: `code/business_entity_resolution/src/package_submission.py`
- Package structure inside `<team_name>_submission.zip`:
  ```
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
  ```

---

## 3. Deliverables & Outputs
- Filled `Documentation_template.md`
- `code/business_entity_resolution/README.md`
- `code/business_entity_resolution/requirements.txt`
- Final submission package archive `<team_name>_submission.zip`

---

## 4. Manual Quality Control (QC) Gate & Stop Instruction

> [!IMPORTANT]
> **STOP & PERFORM FINAL MANUAL QUALITY CONTROL BEFORE PORTAL SUBMISSION**

### Step-by-Step QC Execution Instructions:

1. **Build Final Submission Zip Package**:
   ```bash
   python code/business_entity_resolution/src/package_submission.py --team-name "Tung_Team"
   ```
   *Expected Output*: `Tung_Team_submission.zip` created.

2. **Test Package Unpack & Clean Dry-Run Execution**:
   Perform a clean test extraction in a temporary directory to verify reproducibility:
   ```bash
   mkdir -p scratch/test_unpack
   unzip Tung_Team_submission.zip -d scratch/test_unpack/
   ```
   *Manual Verification Checklist*:
   - [ ] Verify `scratch/test_unpack/output/matching_results.tsv` exists and is non-empty.
   - [ ] Verify `scratch/test_unpack/output/candidate_pairs.tsv` exists and is non-empty.
   - [ ] Verify `scratch/test_unpack/Documentation_template.md` contains filled methodology sections (no template placeholders remaining).
   - [ ] Verify `scratch/test_unpack/code/business_entity_resolution/README.md` contains clear run commands.

3. **Re-Run Official Submission Validator on Unpacked Package**:
   ```bash
   python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
     --matching scratch/test_unpack/output/matching_results.tsv \
     --candidate scratch/test_unpack/output/candidate_pairs.tsv \
     --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
   ```
   *Manual Verification Checklist*:
   - [ ] Confirm output prints **`PASS`**.

4. **Final Checkpoint & Portal Submission Readiness**:
   - [ ] Leaderboard upload file ready: `output/matching_results.tsv`
   - [ ] Final package ready for review: `Tung_Team_submission.zip`

**Final Decision Checkpoint**: Congratulations! The Business Entity Resolution solution package is 100% complete, validated, and ready for official Amazon ML Challenge submission.
