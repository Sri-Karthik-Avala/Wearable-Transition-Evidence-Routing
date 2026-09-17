"""Trimmed adversarial check (one process, no torch): feature-set train-vs-test AUC vs a pseudo-test baseline.
Pseudo baseline = 17 random TRAIN participants vs the other 42 through the same classifier, i.e. the AUC that
participant identifiability alone produces. Excess over it = genuine train->test shift."""
import sys
import time
import numpy as np
import wf_harness as h
import dev_wf_shift_lib as L

t0 = time.time()
d = h.load()
X, Xt, groups = d["X"], d["Xt"], d["groups"]
SETS = {
    "v1like": {"dyn", "clast", "lev", "maglev", "rnorm", "magnm"},
    "nomaglev": {"dyn", "clast", "lev", "rnorm"},
    "center": {"dyn", "clast", "cnorm"},
    "center_inv": {"dyn", "clast", "cnorm", "rnorm", "ang"},
}
names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SETS)
for nm in names:
    bl = SETS[nm]
    A = L.flat(L.site_feats(X, bl)); B = L.flat(L.site_feats(Xt, bl))
    a = L.adv_auc(A, B, rounds=150)
    p, _ = L.pseudo_adv_auc(A, groups, reps=1)
    print(f"{nm:11s} nfeat {A.shape[1]:4d} adv AUC {a:.3f}  pseudo {p:.3f}  excess {a - p:+.3f}  t={time.time() - t0:.0f}s",
          flush=True)
