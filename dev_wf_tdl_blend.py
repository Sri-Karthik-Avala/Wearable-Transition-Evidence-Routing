"""KEY=tdl: numpy-only blend scorer (no torch/pandas -> tiny memory). Blends each tdl OOF with the
LightGBM-per-site reference OOF (oof/tgt__ref_siteReg.npy, 0.7577) using per-row z-scored scores.
usage: python -u dev_wf_tdl_blend.py tdl__name1 tdl__name2 ...
"""
import sys
import numpy as np

z = np.load("wf_cache.npz", allow_pickle=True)
R, lead, groups = z["R"], z["lead"], z["groups"]
G = np.array([1, .7, .45, .25, .1]); D = 1 / np.log2(np.arange(5) + 2.0)
IDEAL, WORST = (G * D).sum(), (G * D[::-1]).sum()


def score(S):
    o = np.argsort(-S, 1, kind="stable")
    return ((np.take_along_axis(R, o, 1) * D).sum(1) - WORST) / (IDEAL - WORST)


def zrow(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)


ref = np.load("oof/tgt__ref_siteReg.npy")
print(f"ref lgb-site {score(ref).mean():.4f}")
for name in sys.argv[1:]:
    S = np.load(f"oof/{name}.npy")
    base = score(S)
    line = [f"{name} alone {base.mean():.4f}"]
    for w in (0.3, 0.5, 0.7):
        line.append(f"w{w} {score(w * zrow(S) + (1 - w) * zrow(ref)).mean():.4f}")
    # paired per-boundary-group difference vs ref (is the gain real?)
    dg = {}
    for g, d in zip(groups, base - score(ref)):
        dg.setdefault(g, []).append(d)
    gm = np.array([np.mean(v) for v in dg.values()])
    line.append(f"vs_ref {gm.mean():+.4f} (group-se {gm.std(ddof=1) / np.sqrt(len(gm)):.4f})")
    print(" | ".join(line), flush=True)
