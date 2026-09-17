import time, warnings
import numpy as np
import lightgbm as lgb
import wf_harness as h
import dev_wf_tab_common as C
warnings.filterwarnings("ignore")

d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
F = h.folds(groups)
L = C.long(X, lead)
n, _, P = L.shape
print("long feats", L.shape, flush=True)

p = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.5, n_jobs=2, verbose=-1, deterministic=True, force_row_wise=True)
t0 = time.time()
oof = np.zeros_like(R)
for trn, val in F:
    m = lgb.LGBMRegressor(**p, random_state=0)
    m.fit(L[trn].reshape(-1, P), R[trn].reshape(-1), categorical_feature=[0])
    oof[val] = m.predict(L[val].reshape(-1, P)).reshape(len(val), 5)
h.report("tab__lgb_long", oof, R, lead)
print(f"  {time.time() - t0:.0f}s se {h.boundary_se(oof, R, groups):.4f}", flush=True)
