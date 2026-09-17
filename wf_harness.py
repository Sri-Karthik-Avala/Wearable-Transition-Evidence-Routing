import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
OOF_DIR = ROOT / "oof"
OOF_DIR.mkdir(exist_ok=True)
SITES = ("RWrist", "RUpArm", "Waist", "LThigh", "LAnkle")
GAINS = np.array([1.0, 0.70, 0.45, 0.25, 0.10], dtype=np.float32)
DISC = 1 / np.log2(np.arange(5) + 2.0)
IDEAL = float((GAINS * DISC).sum())
WORST = float((GAINS * DISC[::-1]).sum())
LEADS = (1.0, 3.0, 5.0, 7.0)


def _relevance(preds):
    R = np.zeros((len(preds), 5), dtype=np.float32)
    for i, p in enumerate(preds):
        for r, s in enumerate(str(p).split(">")):
            R[i, SITES.index(s)] = GAINS[r]
    return R


def load():
    cache = ROOT / "wf_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tr = pd.read_csv(ROOT / "train.csv").merge(pd.read_csv(ROOT / "validation_groups.csv"), on="id")
    te = pd.read_csv(ROOT / "test.csv")
    parse = lambda s: np.stack([np.asarray(json.loads(x), dtype=np.float32) for x in s]).reshape(-1, 8, 45)
    d = dict(X=parse(tr.sensor_sequence), R=_relevance(tr.prediction.values),
             lead=tr.lead_offset_s.values.astype(np.float32), groups=tr.validation_group.values.astype(str),
             Xt=parse(te.sensor_sequence), lead_t=te.lead_offset_s.values.astype(np.float32),
             ids_t=te.id.values.astype(str), pred_str=tr.prediction.values.astype(str))
    np.savez(cache, **d)
    return d


def folds(groups, n=5):
    return list(GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups))


def row_scores(S, R):
    order = np.argsort(-S, axis=1, kind="stable")
    dcg = (np.take_along_axis(R, order, 1) * DISC).sum(1)
    return (dcg - WORST) / (IDEAL - WORST)


def report(name, S, R, lead, save=True):
    sc = row_scores(S, R)
    per = {l: round(float(sc[lead == l].mean()), 4) for l in LEADS}
    print(f"[{name}] OOF {sc.mean():.4f} per-lead {per}", flush=True)
    if save:
        np.save(OOF_DIR / f"{name}.npy", S.astype(np.float32))
    return float(sc.mean())


def boundary_se(S, R, groups, n_boot=2000, n_groups_test=17, seed=0):
    sc = row_scores(S, R)
    g = pd.Series(sc).groupby(groups).mean()
    rng = np.random.default_rng(seed)
    draws = [g.values[rng.integers(0, len(g), n_groups_test)].mean() for _ in range(n_boot)]
    return float(np.std(draws))
