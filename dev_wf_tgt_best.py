"""tgt best member: per-site LightGBM gain regression + kNN over observed ranking classes (expected-gain decode),
per-row z-scored and averaged 50/50. Self-contained: numpy / sklearn / lightgbm only.

fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0) -> (n_te, 5) scores (higher = rank first).
Every test row is scored from its own sensor window + lead; standardisation/PCA are fitted on the training rows only.
"""
import time
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import PCA

GAINS = np.array([1.0, 0.70, 0.45, 0.25, 0.10])
LGB_PRM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
               subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2)
K_NN, N_PCA = 40, 40


def _wide(X, lead):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9)
    f = [Xs.std(1), Xs.mean(1), Xs[:, -1] - Xs[:, 0], np.abs(np.diff(Xs, axis=1)).mean(1),
         Xs[:, -2:].mean(1) - Xs[:, :-2].mean(1), Xs[:, -1]]
    sf = np.concatenate(f, -1).reshape(n, -1)
    return np.concatenate([sf, np.asarray(lead, np.float32)[:, None]], 1).astype(np.float32)


def _z(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0):
    Wtr, Wte = _wide(Xtr, ltr), _wide(Xte, lte)
    Rtr = np.asarray(Rtr, np.float64)
    # member A: per-site gain regression (estimates E[gain_s | x] directly -> Bayes-optimal sort)
    A = np.zeros((len(Xte), 5))
    for s in range(5):
        m = lgb.LGBMRegressor(**LGB_PRM, random_state=seed).fit(Wtr, Rtr[:, s])
        A[:, s] = m.predict(Wte)
    # member B: kNN over ranking classes; p(r) = neighbour frequency, score = sum_r p(r) * gains_r
    mu, sd = Wtr.mean(0), Wtr.std(0) + 1e-6
    pca = PCA(N_PCA, random_state=seed).fit((Wtr - mu) / sd)
    Ztr, Zte = pca.transform((Wtr - mu) / sd), pca.transform((Wte - mu) / sd)
    B = np.zeros((len(Xte), 5))
    for b in range(0, len(Zte), 256):
        d2 = ((Zte[b:b + 256, None, :] - Ztr[None]) ** 2).sum(-1)
        nn = np.argsort(d2, 1, kind="stable")[:, :K_NN]
        B[b:b + 256] = Rtr[nn].mean(1)  # == sum_r p(r) * gain-vector(r)
    return 0.5 * _z(A) + 0.5 * _z(B)


if __name__ == "__main__":
    import wf_harness as h
    d = h.load()
    S = np.zeros((len(d["X"]), 5))
    for tr, va in h.folds(d["groups"]):
        S[va] = fit_predict(d["X"][tr], d["R"][tr], d["lead"][tr], d["X"][va], d["lead"][va], seed=0)
    h.report("tgt__best", S, d["R"], d["lead"])
    print("boundary_se", round(h.boundary_se(S, d["R"], d["groups"]), 4))
    t0 = time.time()
    P = fit_predict(d["X"], d["R"], d["lead"], d["Xt"], d["lead_t"], seed=0)
    print("full-train fit+predict sec", round(time.time() - t0, 1), P.shape)
