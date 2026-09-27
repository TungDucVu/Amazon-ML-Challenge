"""Text preprocessing and normalization for Business Entity Resolution.

Language-agnostic text cleaning that works for US, India, and France (open-set).
Avoids country-specific regexes or hardcoded dictionaries.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set


# Common legal suffixes across countries (language-agnostic list)
LEGAL_SUFFIXES = {
    # English
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "llp", "ltd", "limited", "plc", "lp",
    # India
    "pvt", "private", "nidhi",
    # France
    "sa", "sas", "sarl", "sasu", "sci", "eurl", "snc",
    # General
    "gmbh", "ag", "bv", "nv", "pty",
}


def normalize_text(text: str) -> str:
    """Normalize a text field for comparison.

    Steps:
    1. Unicode NFKD normalization (handles accented chars like é -> e).
    2. Lowercase.
    3. Replace '&' with 'and'.
    4. Strip punctuation (keep alphanumeric and spaces).
    5. Collapse whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, lowercased, normalized string.
    """
    if not text:
        return ""

    # Unicode normalize (decompose accented characters)
    text = unicodedata.normalize("NFKD", text)
    # Remove combining marks (accents) but keep base characters
    text = "".join(c for c in text if not unicodedata.combining(c))

    text = text.lower().strip()

    # Replace '&' with 'and'
    text = text.replace("&", " and ")

    # Replace common punctuation with space
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_tokens(text: str) -> List[str]:
    """Extract word tokens from normalized text.

    Args:
        text: Normalized text string.

    Returns:
        List of word tokens.
    """
    return normalize_text(text).split()


def remove_legal_suffixes(text: str) -> str:
    """Remove legal entity suffixes from a business name.

    Operates on already-normalized text. Language-agnostic.

    Args:
        text: Normalized business name.

    Returns:
        Name with legal suffixes removed.
    """
    tokens = text.split()
    # Remove trailing legal suffixes (can be multiple, e.g. "pvt ltd")
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def extract_numeric_sequences(text: str) -> Set[str]:
    """Extract all numeric sequences (potential postal codes, building numbers).

    Language-agnostic: works for US ZIP (5 digits), Indian PIN (6 digits),
    French postal (5 digits), and building numbers.

    Args:
        text: Raw or normalized text.

    Returns:
        Set of numeric strings found in the text.
    """
    return set(re.findall(r"\b\d{2,6}\b", text))


def create_combined_field(
    business_name: str, business_address: str, country: str
) -> str:
    """Create a single combined text field for embedding/indexing.

    Args:
        business_name: Raw business name.
        business_address: Raw business address.
        country: Country label.

    Returns:
        Combined normalized string: "country name address".
    """
    name_norm = normalize_text(business_name)
    addr_norm = normalize_text(business_address)
    country_norm = normalize_text(country)
    return f"{country_norm} {name_norm} {addr_norm}".strip()


def generate_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Generate character n-grams from text.

    Args:
        text: Input text (should be normalized).
        n: N-gram size (default 3).

    Returns:
        List of character n-gram strings.
    """
    text = normalize_text(text)
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]
