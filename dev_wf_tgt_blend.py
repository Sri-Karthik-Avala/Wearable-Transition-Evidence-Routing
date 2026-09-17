"""Fixed equal-weight blends of saved tgt OOFs (per-row z-scored scores; no weight search)."""
import sys
import numpy as np
import wf_harness as h

d = h.load()
R, lead, groups = d["R"], d["lead"], d["groups"]


def z(S):
    S = S.astype(np.float64)
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


names = sys.argv[1:]
for nm in names:
    S = np.load(f"oof/{nm}.npy")
    print(nm, round(float(h.row_scores(S, R).mean()), 4))
B = sum(z(np.load(f"oof/{nm}.npy")) for nm in names) / len(names)
sc = h.row_scores(B, R)
print("BLEND", round(float(sc.mean()), 4), "se", round(h.boundary_se(B, R, groups), 4))
