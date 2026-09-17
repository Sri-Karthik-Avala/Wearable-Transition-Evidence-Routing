"""tab explorer best member: per-site LightGBM gain regressors on per-row per-site features, 3-seed bag.
Self-contained (numpy + lightgbm). Every feature is computed from a row's own 8x45 window + lead only."""
import time
import numpy as np
import lightgbm as lgb

LEADS = (1.0, 3.0, 5.0, 7.0)
SEEDS = (0, 1, 2)
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, n_jobs=2, verbose=-1, deterministic=True,
              force_row_wise=True)


def _rank(v):
    return np.argsort(np.argsort(v, 1, kind="stable"), 1, kind="stable").astype(np.float32)


def features(X, lead):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).astype(np.float32)
    d = np.diff(Xs, axis=1)
    mu = Xs.mean(1)
    f = {
        "std": Xs.std(1), "mean": mu, "delta": Xs[:, -1] - Xs[:, 0], "absdiff": np.abs(d).mean(1),
        "last": Xs[:, -1], "late": Xs[:, -2:].mean(1) - Xs[:, :-2].mean(1), "lastdev": np.abs(Xs[:, -1] - mu),
        "min": Xs.min(1), "max": Xs.max(1), "lastdiff": d[:, -1], "std_last3": Xs[:, -3:].std(1),
    }
    summ = []
    for k in ("std", "absdiff", "lastdev", "std_last3"):
        v = f[k]
        for c in [v.mean(-1)] + [v[..., 3 * j:3 * j + 3].mean(-1) for j in range(3)]:
            summ += [c, c / (c.mean(1, keepdims=True) + 1e-6), _rank(c)]
    parts = [v.reshape(n, -1) for v in f.values()] + [np.stack(summ, -1).reshape(n, -1), lead[:, None]]
    return np.concatenate(parts, 1).astype(np.float32)


def fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0):
    A, B = features(Xtr, np.asarray(ltr, np.float32)), features(Xte, np.asarray(lte, np.float32))
    out = np.zeros((len(B), 5), dtype=np.float64)
    for s in range(5):
        for sd in SEEDS:
            m = lgb.LGBMRegressor(**PARAMS, random_state=1000 * seed + 100 * sd + s)
            m.fit(A, Rtr[:, s])
            out[:, s] += m.predict(B) / len(SEEDS)
    return out


if __name__ == "__main__":
    import wf_harness as h
    d = h.load()
    X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
    oof = np.zeros_like(R, dtype=np.float64)
    for trn, val in h.folds(groups):
        oof[val] = fit_predict(X[trn], R[trn], lead[trn], X[val], lead[val])
    h.report("tab__best", oof, R, lead)
    print(f"se {h.boundary_se(oof, R, groups):.4f}")
    t0 = time.time()
    St = fit_predict(X, R, lead, d["Xt"], d["lead_t"])
    print(f"full-train fit+predict {time.time() - t0:.1f}s, test scores {St.shape}")
