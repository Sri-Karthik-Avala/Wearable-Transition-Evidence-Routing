"""(1)+(2) adversarial validation train-vs-test per channel group and per feature set; pseudo-test baseline."""
import time
import numpy as np
import wf_harness as h
import dev_wf_shift_lib as L

t0 = time.time()
d = h.load()
X, Xt, groups = d["X"], d["Xt"], d["groups"]
SITES = h.SITES; SENS = ("acc", "gyro", "mag")


def group_feats(X, s, k, center=False):
    V = X[:, :, s * 9 + 3 * k: s * 9 + 3 * k + 3]  # n,8,3
    lev = V.mean(1)
    Vc = V - lev[:, None]
    nr = np.linalg.norm(V, axis=-1)
    dyn = [V.std(1), V[:, -1] - V[:, 0], np.abs(np.diff(V, axis=1)).mean(1)]
    if center:
        nc = np.linalg.norm(Vc, axis=-1)
        return np.concatenate(dyn + [Vc[:, -1], nc.mean(1)[:, None], nc.max(1)[:, None]], 1)
    return np.concatenate([lev] + dyn + [nr.mean(1)[:, None], nr.std(1)[:, None]], 1)


print("== per channel-group adversarial AUC (real test | pseudo-test 17 train participants) ==", flush=True)
rows = []
for s in range(2, 5):  # RWrist + RUpArm already measured in the first (slower) run
    for k in range(3):
        A, B = group_feats(X, s, k), group_feats(Xt, s, k)
        Ac, Bc = group_feats(X, s, k, True), group_feats(Xt, s, k, True)
        Al, Bl = X[:, :, s * 9 + 3 * k: s * 9 + 3 * k + 3].mean(1), Xt[:, :, s * 9 + 3 * k: s * 9 + 3 * k + 3].mean(1)
        a_raw = L.adv_auc(A, B); a_c = L.adv_auc(Ac, Bc); a_l = L.adv_auc(Al, Bl)
        p_raw, _ = L.pseudo_adv_auc(A, groups, reps=1)
        p_c, _ = L.pseudo_adv_auc(Ac, groups, reps=1)
        rows.append((SITES[s], SENS[k], a_raw, p_raw, a_c, p_c, a_l))
        print(f"{SITES[s]:7s} {SENS[k]:4s} raw {a_raw:.3f} (pseudo {p_raw:.3f}) | level-only {a_l:.3f} | "
              f"centered-dyn {a_c:.3f} (pseudo {p_c:.3f})", flush=True)

print(f"time {time.time() - t0:.0f}s", flush=True)
print("== feature-set adversarial AUC (flattened LGBM features) ==", flush=True)
SETS = {
    "v1like": {"dyn", "clast", "lev", "maglev", "rnorm", "magnm"},
    "nomaglev": {"dyn", "clast", "lev", "rnorm"},
    "center": {"dyn", "clast", "cnorm"},
    "center_rnorm": {"dyn", "clast", "cnorm", "rnorm"},
    "inv": {"rnorm", "cnorm", "ang"},
    "center_inv": {"dyn", "clast", "cnorm", "rnorm", "ang"},
    "levels_only": {"lev", "maglev"},
}
for nm, bl in SETS.items():
    A = L.flat(L.site_feats(X, bl)); B = L.flat(L.site_feats(Xt, bl))
    a = L.adv_auc(A, B); p, ps = L.pseudo_adv_auc(A, groups, reps=2)
    print(f"{nm:13s} nfeat {A.shape[1]:4d} adv AUC {a:.3f}  pseudo {p:.3f}+-{ps:.3f}  excess {a - p:+.3f}", flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
