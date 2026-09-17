"""Equal-weight row-z averages of saved OOFs (fixed subsets, no weight search). tab explorer (k-series)."""
import sys
import numpy as np
import wf_harness as h
import dev_wf_tab_common as C

d = h.load(); R, lead, groups = d["R"], d["lead"], d["groups"]
oofs = {p.stem: np.load(p) for p in sorted(h.OOF_DIR.glob("*.npy"))}
oofs = {k: v for k, v in oofs.items() if v.shape == R.shape and np.isfinite(v).all()}
sc = {k: float(h.row_scores(v, R).mean()) for k, v in oofs.items()}
for k in sorted(sc, key=sc.get, reverse=True):
    print(f"{sc[k]:.4f}  {k}")


def blend(names, tag):
    names = [n for n in names if n in oofs]
    if len(names) < 2:
        print("skip", tag, names); return None
    S = np.mean([C.rowz(oofs[n]) for n in names], 0)
    s = h.report(tag, S, R, lead, save=False)
    print(f"   = {names}  se {h.boundary_se(S, R, groups):.4f}", flush=True)
    return s


tab = sorted(k for k in oofs if k.startswith("tab__"))
Z = {k: C.rowz(oofs[k]).ravel() for k in tab}
print("row-z corr (tab members):")
for a in tab:
    print(f"  {a:26s}", " ".join(f"{np.corrcoef(Z[a], Z[b])[0, 1]:.2f}" for b in tab))
for spec in sys.argv[1:]:
    blend(spec.split(","), "blend:" + spec)
