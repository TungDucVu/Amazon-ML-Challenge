"""Post-Processing, Output Formatting & Submission Verification.

Generates the two required output files:
1. output/matching_results.tsv (scored on leaderboard)
2. output/candidate_pairs.tsv (audited for blocking quality)

Enforces:
- Every S1 entity has exactly one row.
- Singletons have empty matched_entity_ids.
- IDs in matching_results.tsv ⊆ IDs in candidate_pairs.tsv.
- No duplicate IDs within a row.
- Only S2-/S3- prefixed IDs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Set


def write_matching_results_tsv(
    predictions: Dict[str, Set[str]],
    output_path: str,
    s1_entity_ids: List[str],
) -> None:
    """Write matching_results.tsv in the exact competition format.

    Args:
        predictions: Dict mapping s1_id -> set of matched entity IDs.
        output_path: Path to write matching_results.tsv.
        s1_entity_ids: Ordered list of all S1 entity IDs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_entity_ids:
            matched = predictions.get(s1_id, set())
            matched_str = ",".join(sorted(matched))
            f.write(f"{s1_id}\t{matched_str}\n")


def verify_subset_constraint(
    predictions: Dict[str, Set[str]],
    candidate_pairs: Dict[str, Set[str]],
) -> List[str]:
    """Verify that all matched IDs are a subset of candidate IDs.

    Returns list of violation messages (empty = no violations).
    """
    violations: List[str] = []
    for s1_id, matched_ids in predictions.items():
        candidates = candidate_pairs.get(s1_id, set())
        extra = matched_ids - candidates
        if extra:
            violations.append(
                f"S1 {s1_id}: matched IDs not in candidates: {sorted(extra)}"
            )
    return violations


def verify_output_format(
    predictions: Dict[str, Set[str]],
    s1_entity_ids: List[str],
) -> List[str]:
    """Verify output format rules before writing files.

    Returns list of error messages (empty = all valid).
    """
    errors: List[str] = []

    # Check every S1 entity has a prediction row
    missing = set(s1_entity_ids) - set(predictions.keys())
    if missing:
        errors.append(f"Missing S1 entities in predictions: {sorted(missing)[:5]}")

    for s1_id, matched_ids in predictions.items():
        for mid in matched_ids:
            # Only S2-/S3- IDs allowed
            if not mid.startswith(("S2-", "S3-")):
                errors.append(f"Invalid ID prefix in {s1_id}: {mid}")

            # No S1- self-matches
            if mid.startswith("S1-"):
                errors.append(f"Self-match S1 ID in {s1_id}: {mid}")

    return errors


def run_official_validator(
    matching_path: str,
    candidate_path: str,
    test_dir: str,
    validator_script: str = "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
) -> Tuple[int, str]:
    """Run the official validate_submission.py script.

    Args:
        matching_path: Path to matching_results.tsv.
        candidate_path: Path to candidate_pairs.tsv.
        test_dir: Path to test dataset directory.
        validator_script: Path to the official validator script.

    Returns:
        Tuple of (exit_code, stdout+stderr output).
    """
    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = result.stdout + result.stderr
    return result.returncode, output


from typing import Tuple
