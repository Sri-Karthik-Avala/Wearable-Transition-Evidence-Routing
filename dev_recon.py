import json, math, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

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

def score_from_scores(S, R):
    order = np.argsort(-S, axis=1, kind="stable")
    dcg = (np.take_along_axis(R, order, 1) * D).sum(1)
    return (dcg - WORST) / (IDEAL - WORST)

tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); vg = pd.read_csv("validation_groups.csv")
tr = tr.merge(vg, on="id")
X = np.stack([np.array(json.loads(s), dtype=np.float32) for s in tr.sensor_sequence]).reshape(-1, 8, 45)
Xt = np.stack([np.array(json.loads(s), dtype=np.float32) for s in te.sensor_sequence]).reshape(-1, 8, 45)
R = rel_matrix(tr.prediction.values)
groups = tr.validation_group.values
print("train", X.shape, "test", Xt.shape, "groups", len(set(groups)))
print(tr.lead_offset_s.value_counts().to_dict(), te.lead_offset_s.value_counts().to_dict())
print(pd.Series(groups).value_counts().describe())
print("top1 site freq", pd.Series([p.split(">")[0] for p in tr.prediction]).value_counts().to_dict())
print("mean rel per site", dict(zip(SITES, R.mean(0).round(3))))
print("n unique rankings", tr.prediction.nunique())
# how consistent are targets across lead offsets of same boundary? (count rows per identical target)
print("frames x-ch stats", X.mean(), X.std(), Xt.mean(), Xt.std())
# baseline: global mean relevance ordering
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, groups=groups))
oof_base = np.zeros(len(X))
for trn, val in folds:
    oof_base[val] = score_from_scores(np.tile(R[trn].mean(0), (len(val), 1)), R[val])
print("prior baseline OOF", oof_base.mean())
# random
rng = np.random.default_rng(0)
print("random", score_from_scores(rng.random(R.shape), R).mean())
# oracle-ish per-site features: std of each site in window, change last vs first
def feats(X, lead):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9)
    f = []
    f.append(Xs.std(1).reshape(n, -1))
    f.append(Xs.mean(1).reshape(n, -1))
    f.append((Xs[:, -1] - Xs[:, 0]).reshape(n, -1))
    f.append(np.abs(np.diff(Xs, axis=1)).mean(1).reshape(n, -1))
    f.append(Xs[:, -1].reshape(n, -1))
    f.append(Xs[:, -2:].mean(1).reshape(n, -1) - Xs[:, :-2].mean(1).reshape(n, -1))
    f.append(lead[:, None])
    return np.concatenate(f, 1)
import lightgbm as lgb
F = feats(X, tr.lead_offset_s.values.astype(np.float32))
oof = np.zeros_like(R)
for trn, val in folds:
    for s in range(5):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
        m.fit(F[trn], R[trn, s])
        oof[val, s] = m.predict(F[val])
sc = score_from_scores(oof, R)
print("LGB per-site OOF", sc.mean(), {l: sc[tr.lead_offset_s.values == l].mean().round(4) for l in [1., 3., 5., 7.]})
