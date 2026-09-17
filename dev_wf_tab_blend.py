"""Equal-weight row-z averages of saved OOFs (no weight search, no greedy selection)."""
import sys
import numpy as np
import wf_harness as h
import dev_wf_tab_common as C

d = h.load(); R, lead, groups = d["R"], d["lead"], d["groups"]
oofs = {p.stem: np.load(p) for p in sorted(h.OOF_DIR.glob("*.npy"))}
sc = {k: float(h.row_scores(v, R).mean()) for k, v in oofs.items()}
for k in sorted(sc, key=sc.get, reverse=True):
    print(f"{sc[k]:.4f}  {k}")


def blend(names, tag):
    names = [n for n in names if n in oofs]
    if len(names) < 2:
        return
    S = np.mean([C.rowz(oofs[n]) for n in names], 0)
    s = h.report(tag, S, R, lead, save=False)
    print(f"   = {names}  se {h.boundary_se(S, R, groups):.4f}")
    return s


# pairwise row-z correlation among tab members
tab = [k for k in oofs if k.startswith("tab__")]
Z = {k: C.rowz(oofs[k]).ravel() for k in tab}
print("row-z corr:")
for i, a in enumerate(tab):
    print(f"  {a:28s}", " ".join(f"{np.corrcoef(Z[a], Z[b])[0, 1]:.2f}" for b in tab))

for spec in sys.argv[1:]:
    blend(spec.split(","), "blend:" + spec)
