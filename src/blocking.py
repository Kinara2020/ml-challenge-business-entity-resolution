import pandas as pd
from src.config import MAX_TOKEN_BUCKET, MIN_SHARED_TOKENS, MAX_CANDIDATES_PER_ENTITY

BLOCK_COLS = ["exact_name", "prefix_name", "postal", "address_token"]
LOOKUP_COLS = ["name_clean", "address_clean", "address_tokens", "postal_code", "country_clean"]

def _build_lookup(df: pd.DataFrame) -> dict:
    """Memory-light replacement for df.set_index(...).to_dict('index'), which
    is extremely heavy at millions of rows. itertuples + zip is far cheaper."""
    lookup = {}
    cols = ["entity_id"] + LOOKUP_COLS
    for row in df[cols].itertuples(index=False, name=None):
        eid = row[0]
        lookup[eid] = dict(zip(LOOKUP_COLS, row[1:]))
    return lookup

class SourceIndex:
    """Precomputed blocking structures for one source (S2 or S3). Build once
    per run, reuse for every S1 chunk."""
    def __init__(self, other_df: pd.DataFrame, label: str):
        self.label = label
        self.lookup = _build_lookup(other_df)
        self.by_name_alnum = other_df.loc[other_df["name_alnum"] != "", ["entity_id", "name_alnum"]]
        self.by_name_prefix = other_df.loc[other_df["name_prefix"] != "", ["entity_id", "name_prefix"]]
        self.by_postal = other_df.loc[other_df["postal_code"] != "", ["entity_id", "postal_code"]]

        o_tok = other_df[["entity_id", "address_tokens"]].explode("address_tokens")
        o_tok = o_tok[o_tok["address_tokens"].notna() & (o_tok["address_tokens"] != "")]
        counts = o_tok["address_tokens"].value_counts()
        self.common_tokens = set(counts[counts > MAX_TOKEN_BUCKET].index)
        self.o_tok = o_tok[~o_tok["address_tokens"].isin(self.common_tokens)]

def _pairs_from_key(s1_df, index_df, key_col, block_name):
    a = s1_df.loc[s1_df[key_col] != "", ["entity_id", key_col]]
    merged = a.merge(index_df, on=key_col, suffixes=("_s1", "_o"))
    out = merged[["entity_id_s1", "entity_id_o"]].drop_duplicates().copy()
    out["block"] = block_name
    return out

def _address_token_pairs(s1_df, idx: "SourceIndex"):
    s1_tok = s1_df[["entity_id", "address_tokens"]].explode("address_tokens")
    s1_tok = s1_tok[s1_tok["address_tokens"].notna() & (s1_tok["address_tokens"] != "")]
    s1_tok = s1_tok[~s1_tok["address_tokens"].isin(idx.common_tokens)]

    merged = s1_tok.merge(idx.o_tok, on="address_tokens", suffixes=("_s1", "_o"))
    pair_counts = merged.groupby(["entity_id_s1", "entity_id_o"]).size()
    strong = pair_counts[pair_counts >= MIN_SHARED_TOKENS].reset_index(name="shared_tokens")
    strong["block"] = "address_token"
    return strong[["entity_id_s1", "entity_id_o", "block", "shared_tokens"]]

def build_candidates(s1_df: pd.DataFrame, idx: "SourceIndex") -> pd.DataFrame:
    blocks = [
        _pairs_from_key(s1_df, idx.by_name_alnum, "name_alnum", "exact_name"),
        _pairs_from_key(s1_df, idx.by_name_prefix, "name_prefix", "prefix_name"),
        _pairs_from_key(s1_df, idx.by_postal, "postal_code", "postal"),
        _address_token_pairs(s1_df, idx),
    ]
    all_blocks = pd.concat(blocks, ignore_index=True)
    if all_blocks.empty:
        cols = ["source1_entity_id", "candidate_entity_id", "n_blocks", "shared_tokens"] \
            + [f"block_{c}" for c in BLOCK_COLS]
        return pd.DataFrame(columns=cols)

    all_blocks["present"] = 1
    flags = all_blocks.pivot_table(
        index=["entity_id_s1", "entity_id_o"], columns="block",
        values="present", aggfunc="max", fill_value=0,
    ).reset_index()
    for col in BLOCK_COLS:
        if col not in flags.columns:
            flags[col] = 0

    shared = (all_blocks[all_blocks["block"] == "address_token"]
              .groupby(["entity_id_s1", "entity_id_o"])["shared_tokens"].max()
              .reset_index())
    pairs = flags.merge(shared, on=["entity_id_s1", "entity_id_o"], how="left")
    pairs["shared_tokens"] = pairs["shared_tokens"].fillna(0)
    pairs["n_blocks"] = pairs[BLOCK_COLS].sum(axis=1)

    pairs = pairs.rename(columns={
        "entity_id_s1": "source1_entity_id", "entity_id_o": "candidate_entity_id",
        **{c: f"block_{c}" for c in BLOCK_COLS},
    })
    return _cap_by_score(pairs)

def _cap_by_score(pairs: pd.DataFrame) -> pd.DataFrame:
    sizes = pairs.groupby("source1_entity_id").size()
    over = sizes[sizes > MAX_CANDIDATES_PER_ENTITY].index
    if len(over) == 0:
        return pairs
    pairs = pairs.copy()
    pairs["_rank_score"] = pairs["n_blocks"] * 1000 + pairs["shared_tokens"]
    capped = (pairs[pairs["source1_entity_id"].isin(over)]
            .sort_values("_rank_score", ascending=False)
            .groupby("source1_entity_id").head(MAX_CANDIDATES_PER_ENTITY))
    kept = pairs[~pairs["source1_entity_id"].isin(over)]
    return pd.concat([kept, capped], ignore_index=True).drop(columns="_rank_score")