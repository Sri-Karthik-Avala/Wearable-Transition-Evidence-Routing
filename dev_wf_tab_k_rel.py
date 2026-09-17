"""Cheap members on shift-robust relative summaries only (ratio to 5-site mean + within-row rank). tab (k-series)."""
import time, warnings
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import wf_harness as h
import dev_wf_tab_common as C
warnings.filterwarnings("ignore")

d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
F = h.folds(groups)
summ = C.site_summary(C.site_feats(X))  # n,5,48 = 16 stats x (raw, ratio, rank)
n = len(X)
rel = np.concatenate([summ[..., 1::3].reshape(n, -1), summ[..., 2::3].reshape(n, -1), C.lead_oh(lead)], 1)
print("rel feats", rel.shape, flush=True)


def cv(make, A, tag):
    t0 = time.time()
    oof = np.zeros_like(R)
    for trn, val in F:
        sc = StandardScaler().fit(A[trn])
        m = make(); m.fit(sc.transform(A[trn]), R[trn]); oof[val] = m.predict(sc.transform(A[val]))
    h.report(tag, oof, R, lead)
    print(f"  {tag} {time.time() - t0:.0f}s", flush=True)


for a in (10.0, 100.0):
    cv(lambda: Ridge(alpha=a), rel, f"tab__k_ridge_rel_a{int(a)}")
for k in (30, 100):
    cv(lambda: KNeighborsRegressor(n_neighbors=k, weights="distance"), rel, f"tab__k_knn_rel_k{k}")
