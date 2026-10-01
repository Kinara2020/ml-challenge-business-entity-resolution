import pandas as pd
import lightgbm as lgb
from src.config import TEST_DIR, OUT_DIR, ROOT, CHUNK_SIZE
from src.utils import load_source
from src.normalize import normalize_df
from src.blocking import SourceIndex, build_candidates
from src.features import compute_features

print("Loading + normalizing test data...")
s1 = normalize_df(load_source(TEST_DIR / "test_source1.tsv"))
s2 = normalize_df(load_source(TEST_DIR / "test_source2.tsv"))
s3 = normalize_df(load_source(TEST_DIR / "test_source3.tsv"))

print("Building blocking indexes (once)...")
idx_s2 = SourceIndex(s2, "S2")
idx_s3 = SourceIndex(s3, "S3")

booster = lgb.Booster(model_file=str(ROOT / "model.txt"))
threshold = float(open(ROOT / "threshold.txt").read())

cand_out, match_out = [], []
n_chunks = (len(s1) // CHUNK_SIZE) + 1

for i in range(n_chunks):
    chunk = s1.iloc[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
    if chunk.empty:
        continue
    print(f"Chunk {i+1}/{n_chunks} ({len(chunk)} S1 entities)...")

    p2 = build_candidates(chunk, idx_s2)
    p3 = build_candidates(chunk, idx_s3)
    lookup1 = chunk.set_index("entity_id").to_dict("index")

    parts = []
    if not p2.empty:
        parts.append(compute_features(p2, lookup1, idx_s2.lookup, "S2"))
    if not p3.empty:
        parts.append(compute_features(p3, lookup1, idx_s3.lookup, "S3"))

    seen_s1 = set()
    if parts:
        feats = pd.concat(parts, ignore_index=True)
        feature_cols = [c for c in feats.columns if c not in ("source1_entity_id", "candidate_entity_id")]
        feats["score"] = booster.predict(feats[feature_cols])

        for s1_id, grp in feats.groupby("source1_entity_id"):
            seen_s1.add(s1_id)
            cand_ids = sorted(set(grp["candidate_entity_id"]))
            match_ids = sorted(set(grp.loc[grp["score"] >= threshold, "candidate_entity_id"]))
            cand_out.append((s1_id, ",".join(cand_ids)))
            match_out.append((s1_id, ",".join(match_ids)))

    for s1_id in chunk["entity_id"]:
        if s1_id not in seen_s1:
            cand_out.append((s1_id, ""))
            match_out.append((s1_id, ""))

pd.DataFrame(cand_out, columns=["source1_entity_id", "candidate_entity_ids"]) \
    .to_csv(OUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)
pd.DataFrame(match_out, columns=["source1_entity_id", "matched_entity_ids"]) \
    .to_csv(OUT_DIR / "matching_results.tsv", sep="\t", index=False)

print(f"Done. {len(match_out)} rows written (expected {len(s1)}).")
print("Run utils/validate_submission.py next.")