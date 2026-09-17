"""KEY=tdl: measure full-train (2104 rows) + test-predict runtime of dev_wf_tdl_best.fit_predict on CPU, 2 threads."""
import time
import numpy as np
import wf_harness as h
from dev_wf_tdl_best import fit_predict, _retry

d = h.load()
t0 = time.time()
S = _retry(lambda: fit_predict(d["X"], d["R"], d["lead"], d["Xt"], d["lead_t"], seed=0))
print(f"full-train runtime {time.time() - t0:.0f}s  test scores {S.shape}  finite {np.isfinite(S).all()}", flush=True)
