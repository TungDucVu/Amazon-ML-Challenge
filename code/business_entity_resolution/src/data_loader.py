import csv
import os
import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def load_tsv(filepath: str, expected_cols: int = None) -> pd.DataFrame:
    """
    Safely loads a tab-separated (.tsv) file with strict type preservation and zero NA conversion.
    
    - Uses sep='\\t' and quoting=csv.QUOTE_NONE (3) to handle quotes/apostrophes cleanly.
    - Uses dtype=str and keep_default_na=False so 'NA', 'N/A', 'None' remain string literals.
    - Validates column count if expected_cols is provided.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        engine="c"
    )

    if expected_cols is not None and df.shape[1] != expected_cols:
        raise ValueError(
            f"Column count mismatch for {filepath}: Expected {expected_cols} columns, got {df.shape[1]}"
        )

    return df


def load_source_file(filepath: str) -> pd.DataFrame:
    """Loads a source file (S1, S2, or S3) ensuring expected 4 columns."""
    df = load_tsv(filepath, expected_cols=4)
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError(f"Header mismatch in {filepath}: Expected {SOURCE_COLUMNS}, got {list(df.columns)}")
    return df


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """Loads a ground truth file ensuring expected 2 columns."""
    df = load_tsv(filepath, expected_cols=2)
    if list(df.columns) != GROUND_TRUTH_COLUMNS:
        raise ValueError(f"Header mismatch in {filepath}: Expected {GROUND_TRUTH_COLUMNS}, got {list(df.columns)}")
    return df
