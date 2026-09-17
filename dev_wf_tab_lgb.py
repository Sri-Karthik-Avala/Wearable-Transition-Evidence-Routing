import sys, time, warnings
import numpy as np
import lightgbm as lgb
import wf_harness as h
import dev_wf_tab_common as C
warnings.filterwarnings("ignore")

d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
F = h.folds(groups)
W = C.wide(X, lead)
print("wide feats", W.shape, flush=True)

BASE = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.5, n_jobs=2, verbose=-1, deterministic=True, force_row_wise=True)
CFG = {
    "ref": {},
    "l7": dict(num_leaves=7, n_estimators=600, learning_rate=0.02, min_child_samples=30, reg_lambda=5.0),
    "d4": dict(max_depth=4, num_leaves=15, n_estimators=500, learning_rate=0.02, min_child_samples=40,
               colsample_bytree=0.3, reg_lambda=10.0),
    "l31": dict(num_leaves=31, colsample_bytree=0.3, min_child_samples=15),
    "xt": dict(extra_trees=True, n_estimators=700, learning_rate=0.03),
}
def run(name, seeds=(0,)):
    t0 = time.time()
    p = {**BASE, **CFG[name]}
    oof = np.zeros_like(R)
    for trn, val in F:
        for s in range(5):
            for sd in seeds:
                m = lgb.LGBMRegressor(**p, random_state=100 * sd + s)
                m.fit(W[trn], R[trn, s]); oof[val, s] += m.predict(W[val]) / len(seeds)
    tag = f"tab__lgb_{name}" + (f"_bag{len(seeds)}" if len(seeds) > 1 else "")
    sc = h.report(tag, oof, R, lead)
    print(f"  {tag} {time.time() - t0:.0f}s se {h.boundary_se(oof, R, groups):.4f}", flush=True)
    return sc


which = sys.argv[1:] or ["ref", "l7"]
res = {nm: run(nm) for nm in which}
best = max(res, key=res.get)
print("best single-seed config:", best, flush=True)
run(best, seeds=(0, 1, 2))
