import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from src.config import (TRAIN_DIR, ROOT, CACHE_DIR, RANDOM_SEED, VAL_FRACTION,THRESHOLD_GRID, CHUNK_SIZE, NEG_PER_POS)
from src.utils import load_source, load_ground_truth, macro_f_beta
from src.normalize import normalize_df
from src.blocking import SourceIndex, build_candidates
from src.features import compute_features

print("Loading + normalizing training data...")
s1 = normalize_df(load_source(TRAIN_DIR / "train_source1.tsv"))
s2 = normalize_df(load_source(TRAIN_DIR / "train_source2.tsv"))
s3 = normalize_df(load_source(TRAIN_DIR / "train_source3.tsv"))
gt = load_ground_truth(TRAIN_DIR / "train_ground_truth.tsv")

print("Building blocking indexes (once)...")
idx_s2 = SourceIndex(s2, "S2")
idx_s3 = SourceIndex(s3, "S3")

gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss.split(s1, groups=s1["entity_id"]))
s1_train = s1.iloc[train_idx].reset_index(drop=True)
s1_val = s1.iloc[val_idx].reset_index(drop=True)

def process_split(s1_split: pd.DataFrame, tag: str, subsample_negatives: bool):
    cand_sizes = []
    total_true_pairs = sum(len(gt.get(s1_id, set())) for s1_id in s1_split["entity_id"])
    total_found = 0
    paths = []
    n_chunks = (len(s1_split) // CHUNK_SIZE) + 1

    for i in range(n_chunks):
        chunk = s1_split.iloc[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        if chunk.empty:
            continue
        t0 = time.time()
        p2 = build_candidates(chunk, idx_s2)
        p3 = build_candidates(chunk, idx_s3)
        if p2.empty and p3.empty:
            continue

        lookup1 = chunk.set_index("entity_id").to_dict("index")
        parts = []
        if not p2.empty:
            parts.append(compute_features(p2, lookup1, idx_s2.lookup, "S2"))
        if not p3.empty:
            parts.append(compute_features(p3, lookup1, idx_s3.lookup, "S3"))
        feats = pd.concat(parts, ignore_index=True)
        feats["label"] = feats.apply(
            lambda r: int(r["candidate_entity_id"] in gt.get(r["source1_entity_id"], set())), axis=1
        )

        sizes = feats.groupby("source1_entity_id").size()
        cand_sizes.extend(sizes.tolist())
        total_found += feats["label"].sum()

        if subsample_negatives:
            pos = feats[feats["label"] == 1]
            neg = feats[feats["label"] == 0].copy()
            keep_n = min(len(neg), NEG_PER_POS * max(len(pos), 1))
            if len(neg) > keep_n:
                neg["_hardness"] = neg["name_token_set"] + neg["address_token_set"]
                neg = neg.sort_values("_hardness", ascending=False).head(keep_n).drop(columns="_hardness")
            feats = pd.concat([pos, neg], ignore_index=True)

        path = CACHE_DIR / f"{tag}_chunk{i}.parquet"
        feats.to_parquet(path, index=False)
        paths.append(path)
        print(f"  [{tag}] chunk {i+1}/{n_chunks}: {len(chunk)} S1, "
              f"{len(feats)} pairs kept, {time.time()-t0:.1f}s")

    ceiling = total_found / max(total_true_pairs, 1)
    arr = np.array(cand_sizes) if cand_sizes else np.array([0])
    print(f"[{tag}] blocking recall ceiling: {total_found}/{total_true_pairs} = {ceiling:.3f}")
    print(f"[{tag}] candidates/S1 — mean {arr.mean():.1f}, median {np.median(arr):.0f}, "
          f"p95 {np.percentile(arr, 95):.0f}, max {arr.max():.0f}")
    return paths

print("Processing TRAIN split (blocking + features, streamed to disk)...")
train_paths = process_split(s1_train, "train", subsample_negatives=True)
print("Processing VAL split (no negative subsampling — need realistic scoring)...")
val_paths = process_split(s1_val, "val", subsample_negatives=False)

train_feats = pd.concat([pd.read_parquet(p) for p in train_paths], ignore_index=True)
val_feats = pd.concat([pd.read_parquet(p) for p in val_paths], ignore_index=True)
print(f"Final training rows: {len(train_feats)}  |  validation rows: {len(val_feats)}")

feature_cols = [c for c in train_feats.columns
                if c not in ("source1_entity_id", "candidate_entity_id", "label")]

model = lgb.LGBMClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.05,
    class_weight="balanced", random_state=RANDOM_SEED,
)
model.fit(train_feats[feature_cols], train_feats["label"])

val_feats = val_feats.copy()
val_feats["score"] = model.predict_proba(val_feats[feature_cols])[:, 1]

print("Searching threshold for best macro F0.5...")
best_thresh, best_f = 0.5, -1.0
for t in THRESHOLD_GRID:
    preds = {}
    for s1_id, grp in val_feats.groupby("source1_entity_id"):
        preds[s1_id] = set(grp.loc[grp["score"] >= t, "candidate_entity_id"])
    score = macro_f_beta(preds, gt, s1_val["entity_id"].tolist())
    if score > best_f:
        best_f, best_thresh = score, t

print(f"Best threshold: {best_thresh}  |  Validation macro F0.5: {best_f:.4f}")

model.booster_.save_model(str(ROOT / "model.txt"))
with open(ROOT / "threshold.txt", "w") as f:
    f.write(str(best_thresh))

fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Top feature importances:\n", fi.head(10))
print("Saved model.txt and threshold.txt")