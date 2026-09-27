"""Multi-Stage Candidate Generation (Blocking) for Business Entity Resolution.

Budget-constrained blocking targeting 5-15 candidates per S1 entity (V2 requirement).

Stages:
1. Exact Country Hard-Blocking Partition (zero recall loss, 66-75% space reduction).
2. Sparse TF-IDF Character 3-gram + Word 1-2 gram cosine retrieval.
3. Generic Numeric/Postal Token Exact-Match Index.
4. Adaptive Dynamic Top-K Filtering with score thresholding.

All stages are language-agnostic and work for unseen countries (France).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from preprocessing import (
    normalize_text,
    remove_legal_suffixes,
    extract_numeric_sequences,
    create_combined_field,
)


def build_tfidf_index(
    texts: List[str],
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 3),
    max_features: int = 50000,
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """Build a TF-IDF index from a list of text strings.

    Args:
        texts: List of text strings to index.
        analyzer: 'char_wb' for character n-grams, 'word' for word n-grams.
        ngram_range: Range of n-gram sizes.
        max_features: Maximum vocabulary size.

    Returns:
        Tuple of (fitted TfidfVectorizer, TF-IDF sparse matrix).
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def sparse_cosine_topk(
    query_matrix: csr_matrix,
    index_matrix: csr_matrix,
    top_k: int = 10,
) -> List[List[Tuple[int, float]]]:
    """Compute top-K cosine similarities between query and index sparse matrices.

    Uses efficient sparse matrix multiplication. Processes in batches to control
    memory for large datasets.

    Args:
        query_matrix: Sparse TF-IDF matrix for queries (n_queries x vocab).
        index_matrix: Sparse TF-IDF matrix for index (n_index x vocab).
        top_k: Number of top results per query.

    Returns:
        List of lists of (index_position, similarity_score) tuples per query.
    """
    batch_size = 500
    n_queries = query_matrix.shape[0]
    results: List[List[Tuple[int, float]]] = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]

        # Sparse dot product = cosine similarity (vectors are L2-normalized by TF-IDF)
        sim_matrix = batch.dot(index_matrix.T)

        for i in range(sim_matrix.shape[0]):
            row = sim_matrix.getrow(i)
            if row.nnz == 0:
                results.append([])
                continue

            indices = row.indices
            scores = row.data

            # Get top-K
            if len(scores) > top_k:
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
            else:
                top_idx = np.argsort(-scores)

            results.append([(int(indices[j]), float(scores[j])) for j in top_idx])

    return results


def build_numeric_token_index(
    df: pd.DataFrame,
) -> Dict[str, Set[int]]:
    """Build an inverted index mapping numeric sequences to row positions.

    Extracts all 2-6 digit sequences from business_address and business_name.
    Works for US ZIP codes, Indian PIN codes, French postal codes, and building numbers.

    Args:
        df: DataFrame with 'business_name' and 'business_address' columns.

    Returns:
        Dict mapping numeric_token -> set of row indices.
    """
    index: Dict[str, Set[int]] = defaultdict(set)
    for idx, row in df.iterrows():
        combined = f"{row['business_name']} {row['business_address']}"
        nums = extract_numeric_sequences(combined)
        for num in nums:
            index[num].add(int(idx))
    return dict(index)


def generate_candidates(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    tfidf_top_k: int = 10,
    similarity_threshold: float = 0.15,
    max_candidates_per_s1: int = 25,
    min_candidates_per_s1: int = 5,
    show_progress: bool = True,
) -> Dict[str, Set[str]]:
    """Generate candidate pairs using multi-stage blocking with country partitioning.

    Implements the V2 budget-constrained pipeline:
    1. Country hard-blocking partition.
    2. TF-IDF character 3-gram cosine retrieval within each partition.
    3. TF-IDF word 1-2 gram cosine retrieval within each partition.
    4. Numeric token exact-match augmentation.
    5. Adaptive score-based filtering + cap enforcement.

    Args:
        df_s1: Source 1 DataFrame (reference entities).
        df_s2: Source 2 DataFrame.
        df_s3: Source 3 DataFrame.
        tfidf_top_k: Number of top TF-IDF candidates per retrieval stage.
        similarity_threshold: Minimum similarity to include a candidate.
        max_candidates_per_s1: Hard ceiling on candidates per S1 entity.
        min_candidates_per_s1: Minimum guaranteed candidates (from top-K).
        show_progress: Show tqdm progress bars.

    Returns:
        Dict mapping source1_entity_id -> set of candidate entity IDs (S2/S3).
    """
    # Combine S2 and S3 into a single target pool
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    # Get unique countries across all sources
    all_countries = set(df_s1["country"].unique()) | set(df_targets["country"].unique())

    candidate_pairs: Dict[str, Set[str]] = {}

    # Initialize all S1 entities with empty candidate sets
    for s1_id in df_s1["entity_id"]:
        candidate_pairs[s1_id] = set()

    for country in sorted(all_countries):
        # Stage 1: Country Hard-Blocking Partition
        s1_country = df_s1[df_s1["country"] == country].reset_index(drop=True)
        targets_country = df_targets[df_targets["country"] == country].reset_index(drop=True)

        if len(s1_country) == 0 or len(targets_country) == 0:
            continue

        if show_progress:
            print(f"  [Blocking] Country={country}: S1={len(s1_country)}, Targets={len(targets_country)}")

        # Prepare combined text fields for TF-IDF
        s1_texts_name = [
            remove_legal_suffixes(normalize_text(row["business_name"]))
            for _, row in s1_country.iterrows()
        ]
        target_texts_name = [
            remove_legal_suffixes(normalize_text(row["business_name"]))
            for _, row in targets_country.iterrows()
        ]

        s1_texts_combined = [
            create_combined_field(row["business_name"], row["business_address"], "")
            for _, row in s1_country.iterrows()
        ]
        target_texts_combined = [
            create_combined_field(row["business_name"], row["business_address"], "")
            for _, row in targets_country.iterrows()
        ]

        # Stage 2: Character 3-gram TF-IDF on business name
        all_name_texts = target_texts_name + s1_texts_name
        char_vectorizer, char_tfidf_all = build_tfidf_index(
            all_name_texts,
            analyzer="char_wb",
            ngram_range=(3, 4),
            max_features=50000,
        )

        n_targets = len(target_texts_name)
        char_tfidf_targets = char_tfidf_all[:n_targets]
        char_tfidf_s1 = char_tfidf_all[n_targets:]

        char_results = sparse_cosine_topk(
            char_tfidf_s1, char_tfidf_targets, top_k=tfidf_top_k
        )

        # Stage 3: Word 1-2 gram TF-IDF on combined name + address
        all_combined_texts = target_texts_combined + s1_texts_combined
        word_vectorizer, word_tfidf_all = build_tfidf_index(
            all_combined_texts,
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50000,
        )

        word_tfidf_targets = word_tfidf_all[:n_targets]
        word_tfidf_s1 = word_tfidf_all[n_targets:]

        word_results = sparse_cosine_topk(
            word_tfidf_s1, word_tfidf_targets, top_k=tfidf_top_k
        )

        # Stage 4: Numeric Token Index
        numeric_index = build_numeric_token_index(targets_country)

        # Merge candidates from all stages for each S1 entity in this country
        for i in range(len(s1_country)):
            s1_id = s1_country.iloc[i]["entity_id"]
            scored_candidates: Dict[int, float] = {}

            # From character n-gram TF-IDF
            for target_idx, score in char_results[i]:
                if target_idx in scored_candidates:
                    scored_candidates[target_idx] = max(scored_candidates[target_idx], score)
                else:
                    scored_candidates[target_idx] = score

            # From word n-gram TF-IDF
            for target_idx, score in word_results[i]:
                if target_idx in scored_candidates:
                    scored_candidates[target_idx] = max(scored_candidates[target_idx], score)
                else:
                    scored_candidates[target_idx] = score

            # From numeric token matching (bonus boost)
            s1_combined = f"{s1_country.iloc[i]['business_name']} {s1_country.iloc[i]['business_address']}"
            s1_nums = extract_numeric_sequences(s1_combined)
            for num in s1_nums:
                if num in numeric_index:
                    for target_idx in numeric_index[num]:
                        if target_idx in scored_candidates:
                            scored_candidates[target_idx] = min(
                                scored_candidates[target_idx] + 0.1, 1.0
                            )
                        else:
                            scored_candidates[target_idx] = 0.1

            # Adaptive dynamic top-K filtering
            if scored_candidates:
                sorted_candidates = sorted(
                    scored_candidates.items(), key=lambda x: -x[1]
                )

                # Always keep at least min_candidates_per_s1 or all if fewer
                selected = []
                for target_idx, score in sorted_candidates:
                    if len(selected) < min_candidates_per_s1:
                        selected.append(target_idx)
                    elif score >= similarity_threshold and len(selected) < max_candidates_per_s1:
                        selected.append(target_idx)
                    else:
                        break

                for target_idx in selected:
                    target_id = targets_country.iloc[target_idx]["entity_id"]
                    candidate_pairs[s1_id].add(target_id)

    return candidate_pairs


def write_candidate_pairs_tsv(
    candidate_pairs: Dict[str, Set[str]],
    output_path: str,
    s1_entity_ids: List[str],
) -> None:
    """Write candidate_pairs.tsv in the exact competition format.

    Args:
        candidate_pairs: Dict mapping s1_id -> set of candidate IDs.
        output_path: Path to write candidate_pairs.tsv.
        s1_entity_ids: Ordered list of all S1 entity IDs (ensures every S1 has a row).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_entity_ids:
            candidates = candidate_pairs.get(s1_id, set())
            candidates_str = ",".join(sorted(candidates))
            f.write(f"{s1_id}\t{candidates_str}\n")
