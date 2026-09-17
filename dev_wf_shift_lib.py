"""Shared helpers for the KEY=shift explorer (features, LGBM-per-site OOF, adversarial AUC, MLP)."""
import warnings
import numpy as np
import lightgbm as lgb

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

EPS = 1e-3


def _tri(Xs, k):
    return Xs[..., 3 * k:3 * k + 3]


def _angles(V):
    """V (n,5,8,3) -> per-site angle stats (n,5,4): mean/max consecutive, first-last, last-vs-mean."""
    nv = np.linalg.norm(V, axis=-1) + EPS
    U = V / nv[..., None]
    cons = np.arccos(np.clip((U[:, :, 1:] * U[:, :, :-1]).sum(-1), -1, 1))
    fl = np.arccos(np.clip((U[:, :, 0] * U[:, :, -1]).sum(-1), -1, 1))
    m = V.mean(2)
    mu = m / (np.linalg.norm(m, axis=-1, keepdims=True) + EPS)
    lm = np.arccos(np.clip((U[:, :, -1] * mu).sum(-1), -1, 1))
    return np.stack([cons.mean(-1), cons.max(-1), fl, lm], -1)


def site_feats(X, blocks):
    """X (n,8,45) -> (n,5,K). blocks: set of names.
    dyn    : per-channel std, last-first, mean|diff|, late-early  (level invariant)
    clast  : centered last frame (last - row mean)                (level invariant)
    lev    : row mean of acc+gyro channels (levels)
    maglev : row mean of mag channels (levels)
    rnorm  : raw-vector norm std/absdiff for acc,gyro,mag (+ acc/gyro norm mean) (orientation invariant)
    magnm  : raw mag-norm mean (field-strength level)
    cnorm  : centered-vector norm mean/max/last for acc,gyro,mag (orientation+level invariant)
    ang    : acc and mag angle stats (orientation change, invariant to fixed rotation)
    """
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)  # n,5,8,9
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
        nr = np.stack([np.linalg.norm(_tri(Xs, k), axis=-1) for k in range(3)], -1)  # n,5,8,3
        if "rnorm" in blocks:
            P += [nr.std(2), np.abs(np.diff(nr, axis=2)).mean(2), nr.mean(2)[..., :2],
                  nr[:, :, -1] - nr.mean(2)]
        if "magnm" in blocks:
            P.append(nr.mean(2)[..., 2:])
    if "cnorm" in blocks:
        nc = np.stack([np.linalg.norm(_tri(Xc, k), axis=-1) for k in range(3)], -1)
        P += [nc.mean(2), nc.max(2), nc[:, :, -1]]
    if "ang" in blocks:
        P += [_angles(_tri(Xs, 0)), _angles(_tri(Xs, 2)), _angles(_tri(Xs, 1))]
    return np.concatenate(P, -1).astype(np.float32)


def flat(Fs, lead=None):
    n = len(Fs)
    rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
    parts = [Fs.reshape(n, -1), rel.reshape(n, -1)]
    if lead is not None:
        parts.append(lead[:, None])
    return np.concatenate(parts, 1).astype(np.float32)


def augment(X, R, lead, rng, copies=1, scale=(0.85, 1.4), shift=0.0, level_scale=False):
    """Train-time aug: per-row per-channel dynamics scaling (+ optional level shift / level scaling)."""
    Xs, Rs, Ls = [X], [R], [lead]
    for _ in range(copies):
        m = X.mean(1, keepdims=True)
        s = rng.uniform(*scale, size=(len(X), 1, 45)).astype(np.float32)
        Xa = m + (X - m) * s
        if level_scale:
            Xa = Xa + m * (rng.uniform(*scale, size=(len(X), 1, 45)).astype(np.float32) - 1)
        if shift > 0:
            Xa = Xa + rng.normal(0, shift, size=(len(X), 1, 45)).astype(np.float32)
        Xs.append(Xa); Rs.append(R); Ls.append(lead)
    return np.concatenate(Xs), np.concatenate(Rs), np.concatenate(Ls)


def lgb_fit_predict(A, Rtr, B, rounds=150, seed=0, n_jobs=2):
    out = np.zeros((len(B), 5), dtype=np.float32)
    for s in range(5):
        m = lgb.LGBMRegressor(n_estimators=rounds, learning_rate=0.06, num_leaves=15, min_child_samples=20,
                              max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              random_state=seed * 10 + s,
                              n_jobs=n_jobs, deterministic=True, force_row_wise=True, verbose=-1)
        for attempt in range(4):  # host commit memory is near its limit (other jobs); retry transient MemoryError
            try:
                m.fit(A, Rtr[:, s])
                out[:, s] = m.predict(B)
                break
            except MemoryError:
                import gc, time as _t
                gc.collect(); _t.sleep(5)
                if attempt == 3:
                    raise
    return out


def lgb_oof(X, R, lead, F, blocks, aug=None, seed=0):
    S = np.zeros_like(R)
    for k, (tr, va) in enumerate(F):
        Xtr, Rtr, ltr = X[tr], R[tr], lead[tr]
        if aug is not None:
            Xtr, Rtr, ltr = augment(Xtr, Rtr, ltr, np.random.default_rng(100 + k), **aug)
        A = flat(site_feats(Xtr, blocks), ltr)
        B = flat(site_feats(X[va], blocks), lead[va])
        S[va] = lgb_fit_predict(A, Rtr, B, seed=seed)
    return S


def adv_auc(A, B, seed=0, rounds=100):
    """Train-vs-test classifier AUC with stratified 5-fold OOF."""
    Z = np.concatenate([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y):
        m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.1, num_leaves=15, min_child_samples=20,
                               max_bin=63,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=seed,
                               n_jobs=2, deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(Z[tr], y[tr])
        p[va] = m.predict_proba(Z[va])[:, 1]
    return roc_auc_score(y, p)


def pseudo_adv_auc(Ftr_flat, groups, n_test_groups=17, reps=3):
    """Identifiability baseline: 17 random TRAIN participants vs the other 42, same classifier."""
    ug = np.unique(groups); out = []
    for r in range(reps):
        rng = np.random.default_rng(r)
        pick = rng.choice(ug, n_test_groups, replace=False)
        msk = np.isin(groups, pick)
        out.append(adv_auc(Ftr_flat[~msk], Ftr_flat[msk], seed=r))
    return float(np.mean(out)), float(np.std(out))
