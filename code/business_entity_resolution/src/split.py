"""Stratified Train/Validation split generator for Business Entity Resolution.

Splits Source 1 entities into 80% Train and 20% Validation stratified by:
1. Country (US, India, etc.)
2. Match cardinality bucket (0 matches/singleton, 1 match, 2+ matches)
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def create_stratified_validation_split(
    df_s1: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    val_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """Create a stratified split of Source 1 entity IDs into train and validation.

    Args:
        df_s1: DataFrame of Source 1 entities with columns ['entity_id', 'country'].
        ground_truth: Dict mapping source1_entity_id -> set of matched IDs.
        val_size: Fraction for validation (default 0.20).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_s1_ids, val_s1_ids).
    """
    df = df_s1.copy()
    s1_ids = df["entity_id"].tolist()

    # Determine cardinality bucket for each entity
    def get_stratum(row: pd.Series) -> str:
        s1_id = row["entity_id"]
        country = str(row["country"]).strip().upper()
        n_matches = len(ground_truth.get(s1_id, set()))
        if n_matches == 0:
            cardinality = "singleton"
        elif n_matches == 1:
            cardinality = "single_match"
        else:
            cardinality = "multi_match"
        return f"{country}__{cardinality}"

    df["stratum"] = df.apply(get_stratum, axis=1)

    # Filter out strata with fewer than 2 elements to allow stratified split
    stratum_counts = df["stratum"].value_counts()
    rare_strata = stratum_counts[stratum_counts < 2].index.tolist()

    if rare_strata:
        df.loc[df["stratum"].isin(rare_strata), "stratum"] = "other"

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["stratum"],
        random_state=random_state,
    )

    train_ids = train_df["entity_id"].tolist()
    val_ids = val_df["entity_id"].tolist()

    return train_ids, val_ids
