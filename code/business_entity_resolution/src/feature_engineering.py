import re
import numpy as np
from rapidfuzz import fuzz, distance

RE_URL = re.compile(r'https?://(?:www\.)?|www\.', re.IGNORECASE)
RE_DOMAINS = re.compile(r'\.(com|in|org|net|co|us|io|biz|info)(?:/.*)?', re.IGNORECASE)
RE_LEGAL = re.compile(r'\b(ltd|limited|inc|incorporated|corp|corporation|llc|llp|pvt|private|co|company|sa|sarl)\b', re.IGNORECASE)
RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_POSTAL = re.compile(r'\b\d{4,6}\b')
RE_HOUSE = re.compile(r'^\d{1,4}\b')
RE_DIGITS = re.compile(r'\d')
RE_NUMERIC_TOKENS = re.compile(r'\b\d+\b')

SOUNDEX_MAP = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6'
}

FEATURE_GROUPS = {
    "string_features": [
        "name_ratio", "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio",
        "name_jaro_winkler", "name_damerau_levenshtein", "name_char_3gram_jaccard",
        "name_char_4gram_jaccard", "name_len_diff_ratio", "clean_name_ratio",
        "clean_name_token_sort_ratio", "clean_name_token_set_ratio", "clean_name_jaro_winkler",
        "legal_suffix_match"
    ],
    "address_features": [
        "addr_ratio", "addr_token_sort_ratio", "addr_token_set_ratio", "addr_jaro_winkler",
        "addr_token_jaccard", "addr_len_diff_ratio", "clean_addr_token_set_ratio",
        "clean_addr_jaro_winkler"
    ],
    "numeric_features": [
        "exact_isolated_postal_match", "house_number_exact_match",
        "digit_jaccard_overlap", "shared_numeric_token_count"
    ],
    "phonetic_features": [
        "soundex_match_ratio"
    ],
    "semantic_features": [
        "semantic_cosine_sim"
    ],
    "blocker_features": [
        "blocker_rank", "blocker_reciprocal_rank"
    ],
    "missingness_features": [
        "is_addr_missing_s1", "is_addr_missing_cand", "is_addr_missing_either",
        "is_name_missing_cand"
    ]
}


def clean_text_advanced(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = text.lower()
    t = RE_URL.sub('', t)
    t = RE_DOMAINS.sub(' ', t)
    t = RE_LEGAL.sub(' ', t)
    t = RE_PUNCT.sub(' ', t)
    return " ".join(t.split())


def extract_legal_suffix(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    m = RE_LEGAL.findall(text.lower())
    return m[-1] if m else ""


def compute_soundex(word: str) -> str:
    if not word or not word.isalpha():
        return ""
    w = word.upper()
    res = [w[0]]
    for ch in w[1:]:
        c = SOUNDEX_MAP.get(ch, '0')
        if c != '0' and c != res[-1]:
            res.append(c)
    return ("".join(res) + "000")[:4]


def extract_entity_metadata(name: str, addr: str) -> dict:
    name_str = name if isinstance(name, str) else ""
    addr_str = addr if isinstance(addr, str) else ""

    clean_n = clean_text_advanced(name_str)
    clean_a = clean_text_advanced(addr_str)

    # Character n-grams for name
    padded = f"  {name_str.lower()}  "
    n3 = set(padded[i:i+3] for i in range(len(padded) - 2)) if len(padded) >= 3 else set()
    n4 = set(padded[i:i+4] for i in range(len(padded) - 3)) if len(padded) >= 4 else set()

    # Address tokens
    addr_toks = set(clean_a.split()) if clean_a else set()

    # Numeric and postal
    postals = set(RE_POSTAL.findall(addr_str))
    h_match = RE_HOUSE.findall(addr_str.strip())
    house = h_match[0] if h_match else ""
    digits = set(RE_DIGITS.findall(addr_str))
    num_tokens = set(RE_NUMERIC_TOKENS.findall(addr_str))

    # Soundex
    soundex_set = set(compute_soundex(w) for w in clean_n.split() if w.isalpha())
    soundex_set.discard("")

    # Legal suffix
    legal = extract_legal_suffix(name_str)

    return {
        "raw_name": name_str,
        "clean_name": clean_n,
        "raw_addr": addr_str,
        "clean_addr": clean_a,
        "name_3grams": n3,
        "name_4grams": n4,
        "addr_toks": addr_toks,
        "postals": postals,
        "house": house,
        "digits": digits,
        "num_tokens": num_tokens,
        "soundex": soundex_set,
        "legal": legal,
        "is_addr_missing": 1 if not addr_str.strip() else 0,
        "is_name_missing": 1 if not name_str.strip() else 0
    }


def compute_pair_features(
    m1: dict,
    m2: dict,
    rank: int,
    vec_sim: float = 0.0
) -> dict:
    n1, n2 = m1["raw_name"], m2["raw_name"]
    cn1, cn2 = m1["clean_name"], m2["clean_name"]
    a1, a2 = m1["raw_addr"], m2["raw_addr"]
    ca1, ca2 = m1["clean_addr"], m2["clean_addr"]

    # --- 1. Raw Name Metrics ---
    f_name_ratio = fuzz.ratio(n1, n2) / 100.0
    f_name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    f_name_tok_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_name_tok_set = fuzz.token_set_ratio(n1, n2) / 100.0
    f_name_jw = distance.JaroWinkler.similarity(n1, n2)
    f_name_dam_lev = distance.DamerauLevenshtein.normalized_similarity(n1, n2)

    g3_1, g3_2 = m1["name_3grams"], m2["name_3grams"]
    f_3gram_jaccard = (len(g3_1.intersection(g3_2)) / len(g3_1.union(g3_2))) if (g3_1 and g3_2) else 0.0

    g4_1, g4_2 = m1["name_4grams"], m2["name_4grams"]
    f_4gram_jaccard = (len(g4_1.intersection(g4_2)) / len(g4_1.union(g4_2))) if (g4_1 and g4_2) else 0.0

    len1, len2 = len(n1), len(n2)
    f_name_len_diff = abs(len1 - len2) / max(len1, len2, 1)

    # --- 2. Clean Name Metrics ---
    f_clean_ratio = fuzz.ratio(cn1, cn2) / 100.0
    f_clean_tok_sort = fuzz.token_sort_ratio(cn1, cn2) / 100.0
    f_clean_tok_set = fuzz.token_set_ratio(cn1, cn2) / 100.0
    f_clean_jw = distance.JaroWinkler.similarity(cn1, cn2)

    leg1, leg2 = m1["legal"], m2["legal"]
    if leg1 and leg2:
        f_legal_match = 1.0 if leg1 == leg2 else 0.0
    else:
        f_legal_match = -1.0

    # --- 3. Address Metrics ---
    addr_miss_s1 = m1["is_addr_missing"]
    addr_miss_c2 = m2["is_addr_missing"]
    addr_miss_either = 1 if (addr_miss_s1 or addr_miss_c2) else 0

    if not addr_miss_either:
        f_addr_ratio = fuzz.ratio(a1, a2) / 100.0
        f_addr_tok_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
        f_addr_tok_set = fuzz.token_set_ratio(a1, a2) / 100.0
        f_addr_jw = distance.JaroWinkler.similarity(a1, a2)
        
        at1, at2 = m1["addr_toks"], m2["addr_toks"]
        f_addr_tok_jaccard = (len(at1.intersection(at2)) / len(at1.union(at2))) if (at1 and at2) else 0.0
        
        alen1, alen2 = len(a1), len(a2)
        f_addr_len_diff = abs(alen1 - alen2) / max(alen1, alen2, 1)

        f_clean_addr_tok_set = fuzz.token_set_ratio(ca1, ca2) / 100.0
        f_clean_addr_jw = distance.JaroWinkler.similarity(ca1, ca2)
    else:
        f_addr_ratio = -1.0
        f_addr_tok_sort = -1.0
        f_addr_tok_set = -1.0
        f_addr_jw = -1.0
        f_addr_tok_jaccard = -1.0
        f_addr_len_diff = -1.0
        f_clean_addr_tok_set = -1.0
        f_clean_addr_jw = -1.0

    # --- 4. Numeric & Postal Features ---
    p1, p2 = m1["postals"], m2["postals"]
    if p1 and p2:
        f_postal_match = 1.0 if bool(p1.intersection(p2)) else 0.0
    else:
        f_postal_match = -1.0

    h1, h2 = m1["house"], m2["house"]
    if h1 and h2:
        f_house_match = 1.0 if h1 == h2 else 0.0
    else:
        f_house_match = -1.0

    d1, d2 = m1["digits"], m2["digits"]
    if d1 and d2:
        f_digit_jaccard = len(d1.intersection(d2)) / len(d1.union(d2))
    elif d1 or d2:
        f_digit_jaccard = 0.0
    else:
        f_digit_jaccard = -1.0

    nt1, nt2 = m1["num_tokens"], m2["num_tokens"]
    f_shared_numeric_count = float(len(nt1.intersection(nt2))) if (nt1 and nt2) else 0.0

    # --- 5. Phonetic Features ---
    s1, s2 = m1["soundex"], m2["soundex"]
    f_soundex_ratio = (len(s1.intersection(s2)) / len(s1.union(s2))) if (s1 and s2) else 0.0

    # --- 6. Blocker Features ---
    f_blocker_rank = float(rank)
    f_blocker_recip_rank = 1.0 / max(1.0, float(rank))

    return {
        # String Features
        "name_ratio": np.float32(np.clip(f_name_ratio, 0.0, 1.0)),
        "name_partial_ratio": np.float32(np.clip(f_name_partial, 0.0, 1.0)),
        "name_token_sort_ratio": np.float32(np.clip(f_name_tok_sort, 0.0, 1.0)),
        "name_token_set_ratio": np.float32(np.clip(f_name_tok_set, 0.0, 1.0)),
        "name_jaro_winkler": np.float32(np.clip(f_name_jw, 0.0, 1.0)),
        "name_damerau_levenshtein": np.float32(np.clip(f_name_dam_lev, 0.0, 1.0)),
        "name_char_3gram_jaccard": np.float32(np.clip(f_3gram_jaccard, 0.0, 1.0)),
        "name_char_4gram_jaccard": np.float32(np.clip(f_4gram_jaccard, 0.0, 1.0)),
        "name_len_diff_ratio": np.float32(np.clip(f_name_len_diff, 0.0, 1.0)),
        "clean_name_ratio": np.float32(np.clip(f_clean_ratio, 0.0, 1.0)),
        "clean_name_token_sort_ratio": np.float32(np.clip(f_clean_tok_sort, 0.0, 1.0)),
        "clean_name_token_set_ratio": np.float32(np.clip(f_clean_tok_set, 0.0, 1.0)),
        "clean_name_jaro_winkler": np.float32(np.clip(f_clean_jw, 0.0, 1.0)),
        "legal_suffix_match": np.float32(f_legal_match),

        # Address Features
        "addr_ratio": np.float32(f_addr_ratio),
        "addr_token_sort_ratio": np.float32(f_addr_tok_sort),
        "addr_token_set_ratio": np.float32(f_addr_tok_set),
        "addr_jaro_winkler": np.float32(f_addr_jw),
        "addr_token_jaccard": np.float32(f_addr_tok_jaccard),
        "addr_len_diff_ratio": np.float32(f_addr_len_diff),
        "clean_addr_token_set_ratio": np.float32(f_clean_addr_tok_set),
        "clean_addr_jaro_winkler": np.float32(f_clean_addr_jw),

        # Numeric Features
        "exact_isolated_postal_match": np.float32(f_postal_match),
        "house_number_exact_match": np.float32(f_house_match),
        "digit_jaccard_overlap": np.float32(f_digit_jaccard),
        "shared_numeric_token_count": np.float32(f_shared_numeric_count),

        # Phonetic Features
        "soundex_match_ratio": np.float32(np.clip(f_soundex_ratio, 0.0, 1.0)),

        # Semantic Feature
        "semantic_cosine_sim": np.float32(np.clip(vec_sim, -1.0, 1.0)),

        # Blocker Features
        "blocker_rank": np.float32(f_blocker_rank),
        "blocker_reciprocal_rank": np.float32(f_blocker_recip_rank),

        # Missingness Indicators
        "is_addr_missing_s1": np.int8(addr_miss_s1),
        "is_addr_missing_cand": np.int8(addr_miss_c2),
        "is_addr_missing_either": np.int8(addr_miss_either),
        "is_name_missing_cand": np.int8(m2["is_name_missing"])
    }
