"""CatBoost MultiRMSE (5 outputs) on the wide per-site features. tab explorer (k-series)."""
import sys, time, warnings
import numpy as np
from catboost import CatBoostRegressor
import wf_harness as h
import dev_wf_tab_common as C
warnings.filterwarnings("ignore")

d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
F = h.folds(groups)
W = C.wide(X, lead)
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 800
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
print("wide", W.shape, iters, depth, flush=True)
t0 = time.time()
oof = np.zeros_like(R, dtype=np.float64)
for i, (trn, val) in enumerate(F):
    m = CatBoostRegressor(loss_function="MultiRMSE", iterations=iters, depth=depth, learning_rate=0.05,
                          l2_leaf_reg=5.0, rsm=0.3, random_seed=0, thread_count=2, verbose=0,
                          border_count=32, used_ram_limit="400mb", allow_writing_files=False)
    m.fit(W[trn], R[trn]); oof[val] = m.predict(W[val])
    print(f"  fold {i} {time.time() - t0:.0f}s", flush=True)
h.report(f"tab__k_cb_i{iters}_d{depth}", oof, R, lead)
print(f"  {time.time() - t0:.0f}s se {h.boundary_se(oof, R, groups):.4f}", flush=True)
