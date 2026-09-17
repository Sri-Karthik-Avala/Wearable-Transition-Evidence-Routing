"""(3a) shift-robust input transforms inside LightGBM-per-site; 5-fold grouped OOF per variant."""
import sys
import time
import numpy as np
import wf_harness as h
import dev_wf_shift_lib as L

d = h.load()
X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
F = h.folds(groups)

SETS = {
    "v1like": {"dyn", "clast", "lev", "maglev", "rnorm", "magnm"},
    "nomaglev": {"dyn", "clast", "lev", "rnorm"},
    "center": {"dyn", "clast", "cnorm"},
    "center_rnorm": {"dyn", "clast", "cnorm", "rnorm"},
    "center_inv": {"dyn", "clast", "cnorm", "rnorm", "ang"},
}
AUGS = {
    "scale": dict(copies=1, scale=(0.85, 1.4)),
    "scaleshift": dict(copies=1, scale=(0.85, 1.4), shift=0.3),
}

which = sys.argv[1] if len(sys.argv) > 1 else "sets"
res = {}
t0 = time.time()
if which == "sets" or which.startswith("sets="):
    pick = which.split("=", 1)[1].split(",") if "=" in which else list(SETS)
    for nm in pick:
        bl = SETS[nm]
        S = L.lgb_oof(X, R, lead, F, bl)
        res[nm] = h.report(f"shift__lgb_{nm}", S, R, lead)
        if nm == "v1like":
            print(f"  boundary_se@17 {h.boundary_se(S, R, groups):.4f}", flush=True)
        print(f"  t={time.time() - t0:.0f}s", flush=True)
else:
    for spec in which.split(","):
        nm, an = spec.split(":")
        S = L.lgb_oof(X, R, lead, F, SETS[nm], aug=AUGS[an])
        res[spec] = h.report(f"shift__lgb_{nm}_{an}", S, R, lead)
        print(f"  t={time.time() - t0:.0f}s", flush=True)
print(res)
