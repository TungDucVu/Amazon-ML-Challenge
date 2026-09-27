"""Language-Agnostic Feature Engineering for Business Entity Resolution.

Computes multi-dimensional similarity features for each (S1, Candidate) pair.
All features are relative similarity ratios (0.0 to 1.0) or language-agnostic
distance metrics so they generalize to unseen countries (France).

Feature Groups:
1. Name String Similarity (Levenshtein, Jaro-Winkler, token ratios)
2. Address String Similarity (token overlap, digit matching)
3. Numeric/Postal Code Matching
4. Legal Suffix Matching
5. Combined TF-IDF Cosine Similarity
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from tqdm import tqdm

from preprocessing import (
    normalize_text,
    remove_legal_suffixes,
    extract_tokens,
    extract_numeric_sequences,
    LEGAL_SUFFIXES,
)


def _safe_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _safe_dice(set_a: set, set_b: set) -> float:
    """Compute Dice coefficient between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return 2 * len(set_a & set_b) / (len(set_a) + len(set_b))


def _char_ngram_set(text: str, n: int = 3) -> set:
    """Generate character n-gram set from text."""
    text = normalize_text(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _get_legal_suffix_status(name: str) -> str:
    """Extract legal suffix status from a business name.

    Returns: 'present' or 'absent'.
    """
    tokens = normalize_text(name).split()
    if tokens and tokens[-1] in LEGAL_SUFFIXES:
        return "present"
    return "absent"


def _get_legal_suffix_tokens(name: str) -> Set[str]:
    """Extract legal suffix tokens from a business name."""
    tokens = normalize_text(name).split()
    suffixes = set()
    for t in reversed(tokens):
        if t in LEGAL_SUFFIXES:
            suffixes.add(t)
        else:
            break
    return suffixes


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
) -> Dict[str, float]:
    """Compute a full feature vector for a single (S1, Candidate) pair.

    Args:
        s1_name: Source 1 business name (raw).
        s1_addr: Source 1 business address (raw).
        s1_country: Source 1 country label.
        cand_name: Candidate business name (raw).
        cand_addr: Candidate business address (raw).
        cand_country: Candidate country label.

    Returns:
        Dict of feature_name -> float feature value.
    """
    features: Dict[str, float] = {}

    # === Normalize ===
    name1 = normalize_text(s1_name)
    name2 = normalize_text(cand_name)
    addr1 = normalize_text(s1_addr)
    addr2 = normalize_text(cand_addr)

    name1_clean = remove_legal_suffixes(name1)
    name2_clean = remove_legal_suffixes(name2)

    # === 1. Name String Similarity Features ===

    # Normalized Levenshtein similarity (on cleaned names)
    features["name_levenshtein_sim"] = (
        Levenshtein.normalized_similarity(name1_clean, name2_clean)
        if name1_clean or name2_clean
        else 1.0
    )

    # Jaro-Winkler similarity
    features["name_jaro_winkler"] = (
        JaroWinkler.similarity(name1_clean, name2_clean)
        if name1_clean or name2_clean
        else 1.0
    )

    # RapidFuzz Token Sort Ratio (handles word reordering)
    features["name_token_sort_ratio"] = fuzz.token_sort_ratio(name1_clean, name2_clean) / 100.0

    # RapidFuzz Token Set Ratio (handles subset matching)
    features["name_token_set_ratio"] = fuzz.token_set_ratio(name1_clean, name2_clean) / 100.0

    # RapidFuzz Partial Ratio (handles substring matching)
    features["name_partial_ratio"] = fuzz.partial_ratio(name1_clean, name2_clean) / 100.0

    # Character 3-gram Jaccard on name
    ngrams1 = _char_ngram_set(name1_clean, 3)
    ngrams2 = _char_ngram_set(name2_clean, 3)
    features["name_char3gram_jaccard"] = _safe_jaccard(ngrams1, ngrams2)

    # Character 4-gram Dice on name
    ngrams1_4 = _char_ngram_set(name1_clean, 4)
    ngrams2_4 = _char_ngram_set(name2_clean, 4)
    features["name_char4gram_dice"] = _safe_dice(ngrams1_4, ngrams2_4)

    # Word-level token Jaccard on name
    name_tokens1 = set(name1_clean.split())
    name_tokens2 = set(name2_clean.split())
    features["name_token_jaccard"] = _safe_jaccard(name_tokens1, name_tokens2)

    # === 2. Address String Similarity Features ===

    # Address Levenshtein similarity
    features["addr_levenshtein_sim"] = (
        Levenshtein.normalized_similarity(addr1, addr2)
        if addr1 or addr2
        else 1.0
    )

    # Address Token Sort Ratio
    features["addr_token_sort_ratio"] = fuzz.token_sort_ratio(addr1, addr2) / 100.0

    # Address Token Set Ratio
    features["addr_token_set_ratio"] = fuzz.token_set_ratio(addr1, addr2) / 100.0

    # Address word-level token Jaccard
    addr_tokens1 = set(addr1.split())
    addr_tokens2 = set(addr2.split())
    features["addr_token_jaccard"] = _safe_jaccard(addr_tokens1, addr_tokens2)

    # Address character 3-gram Jaccard
    addr_ngrams1 = _char_ngram_set(addr1, 3)
    addr_ngrams2 = _char_ngram_set(addr2, 3)
    features["addr_char3gram_jaccard"] = _safe_jaccard(addr_ngrams1, addr_ngrams2)

    # === 3. Numeric Sequence / Postal Code Features ===

    nums1 = extract_numeric_sequences(s1_addr)
    nums2 = extract_numeric_sequences(cand_addr)

    # Numeric Jaccard overlap
    features["numeric_jaccard"] = _safe_jaccard(nums1, nums2)

    # Exact numeric match flag (any shared numeric sequence)
    features["has_numeric_match"] = 1.0 if (nums1 & nums2) else 0.0

    # Number of shared numeric sequences
    features["shared_numeric_count"] = float(len(nums1 & nums2))

    # All-numeric-digits overlap (concatenate all digits, compare)
    digits1 = "".join(sorted(nums1))
    digits2 = "".join(sorted(nums2))
    features["numeric_string_sim"] = (
        Levenshtein.normalized_similarity(digits1, digits2)
        if digits1 or digits2
        else 1.0
    )

    # === 4. Legal Suffix Features ===

    s1_suffixes = _get_legal_suffix_tokens(s1_name)
    cand_suffixes = _get_legal_suffix_tokens(cand_name)

    # Legal suffix match status
    # 1.0 = both same suffix, 0.5 = both absent, 0.0 = mismatch
    if s1_suffixes and cand_suffixes:
        features["legal_suffix_match"] = 1.0 if s1_suffixes == cand_suffixes else 0.0
    elif not s1_suffixes and not cand_suffixes:
        features["legal_suffix_match"] = 0.5
    else:
        features["legal_suffix_match"] = 0.25

    features["legal_suffix_jaccard"] = _safe_jaccard(s1_suffixes, cand_suffixes)

    # === 5. Combined Field Features ===

    # Full combined name + address similarity
    combined1 = f"{name1_clean} {addr1}"
    combined2 = f"{name2_clean} {addr2}"
    features["combined_token_sort_ratio"] = fuzz.token_sort_ratio(combined1, combined2) / 100.0
    features["combined_partial_ratio"] = fuzz.partial_ratio(combined1, combined2) / 100.0

    # === 6. Length and Structural Features ===

    # Name length ratio (language-agnostic structural signal)
    len1 = max(len(name1_clean), 1)
    len2 = max(len(name2_clean), 1)
    features["name_length_ratio"] = min(len1, len2) / max(len1, len2)

    # Token count difference
    features["name_token_count_diff"] = abs(len(name_tokens1) - len(name_tokens2))

    # Country exact match (should always be 1.0 due to blocking, but safety check)
    features["country_match"] = 1.0 if normalize_text(s1_country) == normalize_text(cand_country) else 0.0

    return features


def compute_features_for_pairs(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    candidate_pairs: Dict[str, Set[str]],
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Compute feature matrices for all candidate pairs.

    Args:
        df_s1: Source 1 DataFrame.
        df_targets: Combined S2 + S3 DataFrame.
        candidate_pairs: Dict mapping s1_id -> set of candidate entity IDs.
        show_progress: Show progress bar.

    Returns:
        Tuple of:
        - DataFrame of features (one row per pair).
        - List of (s1_id, candidate_id) tuples in the same order as feature rows.
    """
    # Build lookup dicts for fast access
    s1_lookup: Dict[str, pd.Series] = {}
    for _, row in df_s1.iterrows():
        s1_lookup[row["entity_id"]] = row

    target_lookup: Dict[str, pd.Series] = {}
    for _, row in df_targets.iterrows():
        target_lookup[row["entity_id"]] = row

    all_features: List[Dict[str, float]] = []
    pair_ids: List[Tuple[str, str]] = []

    s1_ids = sorted(candidate_pairs.keys())
    iterator = tqdm(s1_ids, desc="Computing features") if show_progress else s1_ids

    for s1_id in iterator:
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue

        for cand_id in sorted(candidate_pairs[s1_id]):
            cand_row = target_lookup.get(cand_id)
            if cand_row is None:
                continue

            feats = compute_pair_features(
                s1_name=s1_row["business_name"],
                s1_addr=s1_row["business_address"],
                s1_country=s1_row["country"],
                cand_name=cand_row["business_name"],
                cand_addr=cand_row["business_address"],
                cand_country=cand_row["country"],
            )

            all_features.append(feats)
            pair_ids.append((s1_id, cand_id))

    features_df = pd.DataFrame(all_features)
    return features_df, pair_ids
