"""Data loading and schema validation utilities for Amazon ML Challenge 2026.

All source files are strictly tab-separated (.tsv) to avoid issues with commas
in business addresses and match lists.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def load_source_tsv(file_path: str) -> pd.DataFrame:
    """Load a single source TSV file with strict validation.

    Args:
        file_path: Path to the .tsv file.

    Returns:
        DataFrame with columns ['entity_id', 'business_name', 'business_address', 'country'].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file schema is invalid or contains formatting defects.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    # Explicit tab separator, utf-8 encoding, strings for all columns
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    # Validate column schema
    actual_cols = [c.strip().lower() for c in df.columns]
    expected_cols = [c.lower() for c in EXPECTED_SOURCE_COLUMNS]
    if actual_cols != expected_cols:
        raise ValueError(
            f"Invalid columns in {file_path}. Expected {EXPECTED_SOURCE_COLUMNS}, got {list(df.columns)}"
        )

    # Standardize column names
    df.columns = EXPECTED_SOURCE_COLUMNS

    # Strip whitespace and replace NaNs with empty string
    for col in EXPECTED_SOURCE_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def load_ground_truth_tsv(file_path: str) -> Dict[str, Set[str]]:
    """Load train_ground_truth.tsv into a mapping of {source1_id: set_of_matched_ids}.

    Args:
        file_path: Path to train_ground_truth.tsv.

    Returns:
        Dictionary mapping each Source 1 entity_id to a set of matched S2-/S3- entity_ids.
        Singletons map to an empty set().
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    actual_cols = [c.strip().lower() for c in df.columns]
    expected_cols = [c.lower() for c in EXPECTED_GROUND_TRUTH_COLUMNS]
    if actual_cols != expected_cols:
        raise ValueError(
            f"Invalid columns in {file_path}. Expected {EXPECTED_GROUND_TRUTH_COLUMNS}, got {list(df.columns)}"
        )

    df.columns = EXPECTED_GROUND_TRUTH_COLUMNS

    ground_truth: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        matched_raw = row["matched_entity_ids"].strip()
        if matched_raw:
            matched_ids = {m.strip() for m in matched_raw.split(",") if m.strip()}
        else:
            matched_ids = set()
        ground_truth[s1_id] = matched_ids

    return ground_truth


def partition_by_country(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Partition a source DataFrame by the 'country' column.

    This enables exact country hard-blocking, reducing the candidate search space
    by 66-75% without recall loss. Works zero-shot for open-set countries (e.g. France).

    Args:
        df: DataFrame containing at least 'country' column.

    Returns:
        Dict mapping country_name -> subset DataFrame.
    """
    partitions: Dict[str, pd.DataFrame] = {}
    for country, group in df.groupby("country"):
        partitions[str(country)] = group.reset_index(drop=True)
    return partitions


def load_dataset_bundle(
    dir_path: str, is_train: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, Set[str]]]]:
    """Load all 3 sources (and optional ground truth) from a dataset directory.

    Args:
        dir_path: Path to dataset/train or dataset/test.
        is_train: If True, looks for train_source*.tsv and train_ground_truth.tsv.
                  If False, looks for test_source*.tsv.

    Returns:
        Tuple of (df_s1, df_s2, df_s3, ground_truth_dict_or_None).
    """
    prefix = "train" if is_train else "test"
    s1_path = os.path.join(dir_path, f"{prefix}_source1.tsv")
    s2_path = os.path.join(dir_path, f"{prefix}_source2.tsv")
    s3_path = os.path.join(dir_path, f"{prefix}_source3.tsv")

    df_s1 = load_source_tsv(s1_path)
    df_s2 = load_source_tsv(s2_path)
    df_s3 = load_source_tsv(s3_path)

    gt_dict = None
    if is_train:
        gt_path = os.path.join(dir_path, f"{prefix}_ground_truth.tsv")
        if os.path.isfile(gt_path):
            gt_dict = load_ground_truth_tsv(gt_path)

    return df_s1, df_s2, df_s3, gt_dict
