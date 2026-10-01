import pandas as pd
from rapidfuzz import fuzz
import jellyfish

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _digit_overlap(a: str, b: str) -> float:
    da = set(ch for ch in a if ch.isdigit())
    db = set(ch for ch in b if ch.isdigit())
    if not da or not db:
        return 0.0
    return len(da & db) / len(da | db)

def compute_features(pairs: pd.DataFrame, lookup1: dict, lookup2: dict, source_label: str) -> pd.DataFrame:
    rows = []
    for s1_id, o_id, be, bp, bpo, bat, nb, st in zip(
        pairs["source1_entity_id"], pairs["candidate_entity_id"],
        pairs["block_exact_name"], pairs["block_prefix_name"],
        pairs["block_postal"], pairs["block_address_token"],
        pairs["n_blocks"], pairs["shared_tokens"],
    ):
        r1, r2 = lookup1[s1_id], lookup2[o_id]
        n1, n2 = r1["name_clean"], r2["name_clean"]
        a1, a2 = r1["address_clean"], r2["address_clean"]
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": o_id,
            "name_ratio": fuzz.ratio(n1, n2) / 100,
            "name_token_sort": fuzz.token_sort_ratio(n1, n2) / 100,
            "name_token_set": fuzz.token_set_ratio(n1, n2) / 100,
            "name_partial": fuzz.partial_ratio(n1, n2) / 100,
            "name_jaro": jellyfish.jaro_winkler_similarity(n1, n2) if n1 and n2 else 0.0,
            "address_ratio": fuzz.ratio(a1, a2) / 100,
            "address_token_set": fuzz.token_set_ratio(a1, a2) / 100,
            "address_jaccard": _jaccard(r1["address_tokens"], r2["address_tokens"]),
            "postal_match": int(bool(r1["postal_code"]) and r1["postal_code"] == r2["postal_code"]),
            "country_match": int(r1["country_clean"] == r2["country_clean"] and r1["country_clean"] != ""),
            "digit_overlap": _digit_overlap(a1, a2),
            "name_len_diff": abs(len(n1) - len(n2)),
            "address_len_diff": abs(len(a1) - len(a2)),
            "block_exact_name": be,
            "block_prefix_name": bp,
            "block_postal": bpo,
            "block_address_token": bat,
            "n_blocks": nb,
            "shared_tokens_block": st,
            "source_is_s2": int(source_label == "S2"),
            "source_is_s3": int(source_label == "S3"),
        })
    return pd.DataFrame(rows)