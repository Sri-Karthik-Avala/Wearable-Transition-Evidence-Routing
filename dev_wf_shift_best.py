"""KEY=shift best member: LightGBM-per-site gain regression on shift-robust per-site features.
Self-contained: fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0) -> (n_te,5) expected-gain scores.
Every test row is featurised from its own 8x45 window + lead only (per-row self-normalisation only)."""
import time
import numpy as np
import lightgbm as lgb

BLOCKS = {"dyn", "clast", "lev", "maglev", "rnorm", "magnm"}  # set after the transform sweep
AUG = None  # e.g. dict(scale=(0.85, 1.4)) -> one dynamics-rescaled copy of the train rows
ROUNDS = 150
N_JOBS = 2
EPS = 1e-3


def _tri(Xs, k):
    return Xs[..., 3 * k:3 * k + 3]


def _angles(V):
    nv = np.linalg.norm(V, axis=-1) + EPS
    U = V / nv[..., None]
    cons = np.arccos(np.clip((U[:, :, 1:] * U[:, :, :-1]).sum(-1), -1, 1))
    fl = np.arccos(np.clip((U[:, :, 0] * U[:, :, -1]).sum(-1), -1, 1))
    m = V.mean(2)
    mu = m / (np.linalg.norm(m, axis=-1, keepdims=True) + EPS)
    lm = np.arccos(np.clip((U[:, :, -1] * mu).sum(-1), -1, 1))
    return np.stack([cons.mean(-1), cons.max(-1), fl, lm], -1)


def site_feats(X, blocks):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)
    Xc = Xs - Xs.mean(2, keepdims=True)
    P = []
    if "dyn" in blocks:
        P += [Xs.std(2), Xs[:, :, -1] - Xs[:, :, 0], np.abs(np.diff(Xs, axis=2)).mean(2),
              Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2)]
    if "clast" in blocks:
        P.append(Xc[:, :, -1])
    if "lev" in blocks:
        P.append(Xs.mean(2)[..., :6])
    if "maglev" in blocks:
        P.append(Xs.mean(2)[..., 6:])
    if "rnorm" in blocks or "magnm" in blocks:
        nr = np.stack([np.linalg.norm(_tri(Xs, k), axis=-1) for k in range(3)], -1)
        if "rnorm" in blocks:
            P += [nr.std(2), np.abs(np.diff(nr, axis=2)).mean(2), nr.mean(2)[..., :2], nr[:, :, -1] - nr.mean(2)]
        if "magnm" in blocks:
            P.append(nr.mean(2)[..., 2:])
    if "cnorm" in blocks:
        nc = np.stack([np.linalg.norm(_tri(Xc, k), axis=-1) for k in range(3)], -1)
        P += [nc.mean(2), nc.max(2), nc[:, :, -1]]
    if "ang" in blocks:
        P += [_angles(_tri(Xs, 0)), _angles(_tri(Xs, 2)), _angles(_tri(Xs, 1))]
    return np.concatenate(P, -1).astype(np.float32)


def flat(Fs, lead):
    n = len(Fs)
    rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
    return np.concatenate([Fs.reshape(n, -1), rel.reshape(n, -1), lead[:, None]], 1).astype(np.float32)


def fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0):
    Xtr = np.asarray(Xtr, np.float32); Xte = np.asarray(Xte, np.float32)
    ltr = np.asarray(ltr, np.float32); lte = np.asarray(lte, np.float32)
    if AUG is not None:
        rng = np.random.default_rng(seed + 100)
        m = Xtr.mean(1, keepdims=True)
        s = rng.uniform(*AUG["scale"], size=(len(Xtr), 1, 45)).astype(np.float32)
        Xtr = np.concatenate([Xtr, m + (Xtr - m) * s]); Rtr = np.concatenate([Rtr, Rtr]); ltr = np.concatenate([ltr, ltr])
    A = flat(site_feats(Xtr, BLOCKS), ltr)
    B = flat(site_feats(Xte, BLOCKS), lte)
    out = np.zeros((len(B), 5), dtype=np.float32)
    for s in range(5):
        mdl = lgb.LGBMRegressor(n_estimators=ROUNDS, learning_rate=0.06, num_leaves=15, min_child_samples=20,
                                max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=seed * 10 + s,
                                n_jobs=N_JOBS, deterministic=True, force_row_wise=True, verbose=-1)
        mdl.fit(A, Rtr[:, s])
        out[:, s] = mdl.predict(B)
    return out


if __name__ == "__main__":
    import wf_harness as h
    d = h.load()
    X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
    S = np.zeros_like(R)
    for tr, va in h.folds(groups):
        S[va] = fit_predict(X[tr], R[tr], lead[tr], X[va], lead[va], seed=0)
    h.report("shift__best", S, R, lead)
    print(f"boundary_se@17 {h.boundary_se(S, R, groups):.4f}", flush=True)
    t0 = time.time()
    St = fit_predict(X, R, lead, d["Xt"], d["lead_t"], seed=0)
    print(f"full-train fit+predict runtime {time.time() - t0:.1f}s  test pred shape {St.shape}", flush=True)
