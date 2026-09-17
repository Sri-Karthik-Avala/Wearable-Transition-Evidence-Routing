"""Long-format LightGBM (one row per window x site, site id categorical, target gain) with
own-minus-other-sites relative features. tab explorer (k-series)."""
import sys, time, warnings
import numpy as np
import lightgbm as lgb
import wf_harness as h
import dev_wf_tab_common as C
warnings.filterwarnings("ignore")

d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
F = h.folds(groups)


def long_rel(X, lead):
    f = C.site_feats(X)
    n = len(X)
    keys = ["std", "absdiff", "lastdev", "std_last3", "late", "delta", "lastdiff"]
    per = np.concatenate([f[k] for k in keys], -1)  # n,5,63
    rel = per - per.mean(1, keepdims=True)  # own minus 5-site mean, per channel-feature
    summ = C.site_summary(f)  # n,5,48
    ctx = np.concatenate([summ[..., 0::12].reshape(n, -1), f["mean"].reshape(n, -1)], 1)  # site means of key summaries + activity signature
    rows = []
    for s in range(5):
        sid = np.full((n, 1), s, np.float32)
        rows.append(np.concatenate([sid, lead[:, None], per[:, s], rel[:, s], summ[:, s], f["mean"][:, s], ctx], 1))
    return np.stack(rows, 1).astype(np.float32)


L = long_rel(X, lead)
n, _, P = L.shape
print("long_rel feats", L.shape, flush=True)
p = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0, n_jobs=2, verbose=-1, deterministic=True,
         force_row_wise=True)
t0 = time.time()
oof = np.zeros_like(R)
for trn, val in F:
    m = lgb.LGBMRegressor(**p, random_state=0)
    m.fit(L[trn].reshape(-1, P), R[trn].reshape(-1), categorical_feature=[0])
    oof[val] = m.predict(L[val].reshape(-1, P)).reshape(len(val), 5)
    print(f"  fold {time.time() - t0:.0f}s", flush=True)
h.report("tab__k_long_rel", oof, R, lead)
print(f"  {time.time() - t0:.0f}s se {h.boundary_se(oof, R, groups):.4f}", flush=True)
