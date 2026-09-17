import sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import dev_sol_v2 as S

G = S.GAINS; D = 1 / np.log2(np.arange(5) + 2.0)
IDEAL = (G * D).sum(); WORST = (G * D[::-1]).sum()
def score(Sc, R):
    order = np.argsort(-Sc, 1, kind="stable")
    return ((np.take_along_axis(R, order, 1) * D).sum(1) - WORST) / (IDEAL - WORST)

tr = pd.read_csv("train.csv").merge(pd.read_csv("validation_groups.csv"), on="id")
X = S.parse_seq(tr.sensor_sequence); R = S.relevance(tr.prediction.values)
l = tr.lead_offset_s.values.astype(np.float32)
nf = int(sys.argv[1]) if len(sys.argv) > 1 else 5
folds = list(GroupKFold(5).split(X, groups=tr.validation_group.values))[:nf]
idx = np.concatenate([v for _, v in folds])
ob, on, ol = np.zeros_like(R), np.zeros_like(R), np.zeros_like(R)
for k, (a, b) in enumerate(folds):
    ob[b], on[b], ol[b] = S.fit_predict(X[a], R[a], l[a], X[b], l[b])
    print(k, "blend", score(ob[b], R[b]).mean().round(4), "nn", score(on[b], R[b]).mean().round(4), "lgb", score(ol[b], R[b]).mean().round(4), flush=True)
z = lambda A: (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-6)
for w in [0, 0.3, 0.5, 0.7, 1.0]:
    print("w_nn", w, score(w * z(on[idx]) + (1 - w) * z(ol[idx]), R[idx]).mean().round(4))
np.save("dev_oof_v2.npy", np.stack([on, ol]))
