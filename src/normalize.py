"""Vectorized normalization — uses pandas .str methods, not row-by-row apply,
so it stays fast at 5M+ rows."""
import pandas as pd
from src.config import NAME_PREFIX_LEN, MIN_TOKEN_LEN

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt", "ltd", "limited", "corporation",
    "corp", "inc", "incorporated", "llc", "llp", "co", "company",
    "enterprises", "industries", "group", "gmbh", "sarl", "sas",
]
ABBREV_MAP = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bapt\b": "apartment", r"\bfl\b": "floor",
    r"\bnr\b": "near", r"\bopp\b": "opposite", r"&": "and",
}
SUFFIX_PATTERN = r"\b(" + "|".join(LEGAL_SUFFIXES) + r")\b"

def _clean_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    s = s.str.replace(r"[^\w\s&]", " ", regex=True)
    for pat, rep in ABBREV_MAP.items():
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name_clean"] = _clean_series(df["business_name"])
    df["address_clean"] = _clean_series(df["business_address"])
    df["country_clean"] = df["country"].fillna("").str.lower().str.strip()

    df["name_core"] = df["name_clean"].str.replace(SUFFIX_PATTERN, "", regex=True).str.strip()
    df["name_alnum"] = df["name_core"].str.replace(r"[^a-z0-9]", "", regex=True)
    df["name_prefix"] = df["name_alnum"].str[:NAME_PREFIX_LEN]

    df["address_tokens"] = df["address_clean"].apply(
        lambda s: frozenset(t for t in s.split() if len(t) >= MIN_TOKEN_LEN)
    )
    df["postal_code"] = df["address_clean"].str.extractall(r"(\d{4,6})").groupby(level=0)[0].last()
    df["postal_code"] = df["postal_code"].fillna("")
    return df