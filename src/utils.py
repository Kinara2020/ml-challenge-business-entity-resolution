import pandas as pd

def load_source(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")

def load_ground_truth(path):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    gt = {}
    for s1, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        gt[s1] = set(x for x in matched.split(",") if x)
    return gt

def f_beta_row(pred: set, true: set, beta=0.5) -> float:
    if not true and not pred:
        return 1.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true) if true else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0

def macro_f_beta(preds: dict, gt: dict, all_s1_ids, beta=0.5) -> float:
    scores = [f_beta_row(preds.get(s1, set()), gt.get(s1, set()), beta) for s1 in all_s1_ids]
    return sum(scores) / len(scores)