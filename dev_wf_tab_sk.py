"""Cheap sklearn members on the wide features: ExtraTrees multi-output, ridge, kNN (standardized, fold-fitted)."""
import time, warnings
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import wf_harness as h
import dev_wf_tab_common as C
warnings.filterwarnings("ignore")

d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
F = h.folds(groups)
W = np.concatenate([C.wide(X, lead), C.lead_oh(lead)], 1)
Sm = np.concatenate([C.summary_only(X, lead), C.lead_oh(lead)], 1)


def cv(make, A, tag, scale=False):
    t0 = time.time()
    oof = np.zeros_like(R)
    for trn, val in F:
        a, b = A[trn], A[val]
        if scale:
            sc = StandardScaler().fit(a); a, b = sc.transform(a), sc.transform(b)
        m = make(); m.fit(a, R[trn]); oof[val] = m.predict(b)
    h.report(tag, oof, R, lead)
    print(f"  {tag} {time.time() - t0:.0f}s", flush=True)


cv(lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.3, n_jobs=2, random_state=0),
   W, "tab__et")
for a in (30.0, 300.0):
    cv(lambda: Ridge(alpha=a), W, f"tab__ridge_a{int(a)}", scale=True)
for k in (25, 75):
    cv(lambda: KNeighborsRegressor(n_neighbors=k, weights="distance"), Sm, f"tab__knn_summ_k{k}", scale=True)
