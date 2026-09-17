import json, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
SITES = ("RWrist", "RUpArm", "Waist", "LThigh", "LAnkle")
G = np.array([1.0, 0.70, 0.45, 0.25, 0.10])
D = 1 / np.log2(np.arange(5) + 2.0)
IDEAL = (G * D).sum(); WORST = (G * D[::-1]).sum()

def rel_matrix(preds):
    R = np.zeros((len(preds), 5))
    for i, p in enumerate(preds):
        for r, s in enumerate(p.split(">")):
            R[i, SITES.index(s)] = G[r]
    return R

def score(S, R):
    order = np.argsort(-S, axis=1, kind="stable")
    dcg = (np.take_along_axis(R, order, 1) * D).sum(1)
    return (dcg - WORST) / (IDEAL - WORST)

tr = pd.read_csv("train.csv").merge(pd.read_csv("validation_groups.csv"), on="id")
X = np.stack([np.array(json.loads(s), dtype=np.float32) for s in tr.sensor_sequence]).reshape(-1, 8, 45)
R = rel_matrix(tr.prediction.values)
lead = tr.lead_offset_s.values.astype(np.float32)
groups = tr.validation_group.values
folds = list(GroupKFold(5).split(X, groups=groups))
n = len(X)
Xs = X.reshape(n, 8, 5, 9)

def base_site_feats(Xs):
    f = {}
    f["std"] = Xs.std(1)
    f["mean"] = Xs.mean(1)
    f["delta"] = Xs[:, -1] - Xs[:, 0]
    f["absdiff"] = np.abs(np.diff(Xs, axis=1)).mean(1)
    f["last"] = Xs[:, -1]
    f["late"] = Xs[:, -2:].mean(1) - Xs[:, :-2].mean(1)
    return f  # each (n,5,9)

def rich_site_feats(Xs):
    f = base_site_feats(Xs)
    nrm = np.stack([np.linalg.norm(Xs[..., k*3:(k+1)*3], axis=-1) for k in range(3)], -1)  # n,8,5,3
    f["nrm_mean"] = nrm.mean(1); f["nrm_std"] = nrm.std(1)
    f["nrm_last"] = nrm[:, -1] - nrm.mean(1)
    f["nrm_absdiff"] = np.abs(np.diff(nrm, axis=1)).mean(1)
    f["nrm_min"] = nrm.min(1); f["nrm_max"] = nrm.max(1)
    f["lastdev"] = np.abs(Xs[:, -1] - Xs.mean(1))
    f["min"] = Xs.min(1); f["max"] = Xs.max(1)
    return f

def site_summary(f):
    # per-site scalar summaries across channels (n,5,k)
    out = []
    for k in ["std", "absdiff", "lastdev", "nrm_std", "nrm_absdiff"]:
        if k in f:
            v = f[k].mean(-1)  # n,5
            out.append(v)
            out.append(v / (v.mean(1, keepdims=True) + 1e-6))
            out.append(np.argsort(np.argsort(v, 1), 1).astype(np.float32))
    return np.stack(out, -1)

def wide(f, extra=True):
    parts = [v.reshape(n, -1) for v in f.values()]
    if extra:
        parts.append(site_summary(f).reshape(n, -1))
    parts.append(lead[:, None])
    return np.concatenate(parts, 1)

def run_wide(F, tag):
    oof = np.zeros_like(R)
    for trn, val in folds:
        for s in range(5):
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=4)
            m.fit(F[trn], R[trn, s]); oof[val, s] = m.predict(F[val])
    sc = score(oof, R)
    print(tag, F.shape[1], round(sc.mean(), 4), {l: round(sc[lead == l].mean(), 4) for l in [1., 3., 5., 7.]}, flush=True)
    return oof

def long_data(f):
    per = np.concatenate([v for v in f.values()], -1)  # n,5,K
    summ = site_summary(f)  # n,5,S
    rows = []
    for s in range(5):
        own = np.concatenate([per[:, s], summ[:, s]], -1)
        others = per.reshape(n, -1)
        sid = np.full((n, 1), s, np.float32)
        rows.append(np.concatenate([sid, lead[:, None], own, summ.reshape(n, -1), others], 1))
    return np.stack(rows, 1)  # n,5,F

def run_long(L, tag, rank=False):
    oof = np.zeros_like(R)
    lab = np.zeros_like(R, dtype=int)
    for i in range(5):
        lab[:, :] = np.round(np.searchsorted(-G, -R) ).astype(int)
    labint = (4 - np.searchsorted(-G, -R)).astype(int)  # 4 for top
    for trn, val in folds:
        Xtr = L[trn].reshape(-1, L.shape[2]); Xva = L[val].reshape(-1, L.shape[2])
        if rank:
            m = lgb.LGBMRanker(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=4, label_gain=[0.1, 0.25, 0.45, 0.7, 1.0])
            m.fit(Xtr, labint[trn].reshape(-1), group=np.full(len(trn), 5), categorical_feature=[0])
        else:
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=4)
            m.fit(Xtr, R[trn].reshape(-1), categorical_feature=[0])
        oof[val] = m.predict(Xva).reshape(len(val), 5)
    sc = score(oof, R)
    print(tag, L.shape[2], round(sc.mean(), 4), {l: round(sc[lead == l].mean(), 4) for l in [1., 3., 5., 7.]}, flush=True)
    return oof

which = sys.argv[1]
fb = base_site_feats(Xs); fr = rich_site_feats(Xs)
if which == "wide":
    run_wide(wide(fb, extra=False), "wide_base")
    run_wide(wide(fb, extra=True), "wide_base+summ")
    run_wide(wide(fr, extra=True), "wide_rich+summ")
elif which == "long":
    o1 = run_long(long_data(fr), "long_reg_rich")
    o2 = run_long(long_data(fr), "long_rank_rich", rank=True)
    sc = score(o1 / o1.std() + o2 / o2.std(), R); print("long blend", round(sc.mean(), 4))
