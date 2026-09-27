import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

# Copy some preprocessing logic
import re
import unicodedata

LEGAL_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp", "ltd", "limited", "plc", "lp", "pvt", "private", "nidhi", "sa", "sas", "sarl", "sasu", "sci", "eurl", "snc", "gmbh", "ag", "bv", "nv", "pty"}

GENERIC_TOKENS = {"global", "services", "india", "corporation", "inc", "ltd", "private", "company", "technologies", "systems", "solutions", "group", "enterprises"}

def normalize_text(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_legal_suffixes(text: str) -> str:
    tokens = text.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)

def extract_numeric_sequences(text: str):
    return set(re.findall(r"\b\d{2,6}\b", text))

def compute_targeted_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country):
    features = {}
    
    name1 = normalize_text(s1_name)
    name2 = normalize_text(cand_name)
    addr1 = normalize_text(s1_addr)
    addr2 = normalize_text(cand_addr)

    name1_clean = remove_legal_suffixes(name1)
    name2_clean = remove_legal_suffixes(name2)
    
    # EXACT NORMALIZED NAME
    features['exact_name_match'] = 1.0 if name1_clean == name2_clean and name1_clean else 0.0
    
    # DISTINCTIVE TOKEN OVERLAP (Name)
    tokens1 = set(name1_clean.split()) - GENERIC_TOKENS
    tokens2 = set(name2_clean.split()) - GENERIC_TOKENS
    if tokens1 and tokens2:
        features['distinctive_name_jaccard'] = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    else:
        features['distinctive_name_jaccard'] = 0.0
        
    # SHARED NON-GENERIC TOKENS (Count)
    features['shared_distinctive_tokens'] = float(len(tokens1 & tokens2))
    
    # NAME CONTAINMENT
    if tokens1 and tokens2:
        features['name_containment'] = 1.0 if tokens1.issubset(tokens2) or tokens2.issubset(tokens1) else 0.0
    else:
        features['name_containment'] = 0.0
        
    # EXACT NUMERIC TOKEN AGREEMENT
    nums1 = extract_numeric_sequences(s1_addr)
    nums2 = extract_numeric_sequences(cand_addr)
    
    if nums1 and nums2:
        features['exact_numeric_agreement'] = 1.0 if nums1 == nums2 else 0.0
    elif not nums1 and not nums2:
        features['exact_numeric_agreement'] = 0.5 # Missing both
    else:
        features['exact_numeric_agreement'] = 0.0
        
    # CROSS FIELD
    name_sim = fuzz.token_sort_ratio(name1_clean, name2_clean) / 100.0
    addr_sim = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    features['both_strong_agree'] = 1.0 if name_sim > 0.8 and addr_sim > 0.8 else 0.0
    features['name_agree_addr_disagree'] = 1.0 if name_sim > 0.8 and addr_sim < 0.4 else 0.0
    features['addr_agree_name_disagree'] = 1.0 if addr_sim > 0.8 and name_sim < 0.4 else 0.0
    
    return features
