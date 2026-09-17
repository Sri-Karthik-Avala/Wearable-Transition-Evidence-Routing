"""KEY=tdl: LightGBM-per-site reference OOF (copy of shipped v1 member, n_jobs=2) for blend tests."""
import numpy as np
import lightgbm as lgb
import wf_harness as h


def site_stats(X):  # local copy: importing dev_wf_tdl_x would pull in torch (~1 GB commit)
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)
    norms = np.stack([np.linalg.norm(Xs[..., k:k + 3], axis=-1) for k in (0, 3, 6)], -1)
    parts = [Xs.std(2), Xs.mean(2), Xs[:, :, -1] - Xs[:, :, 0], np.abs(np.diff(Xs, axis=2)).mean(2),
             Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2), norms.std(2), norms.mean(2),
             np.abs(np.diff(norms, axis=2)).mean(2)]
    return np.concatenate(parts, -1).astype(np.float32)


def fit_predict_lgb(Xtr, Rtr, ltr, Xte, lte):
    Ftr, Fte = site_stats(Xtr), site_stats(Xte)

    def flat(Fs, X, l):
        n = len(Fs)
        rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
        return np.concatenate([Fs.reshape(n, -1), rel.reshape(n, -1), X[:, -1], l[:, None]], 1)
    A, B = flat(Ftr, Xtr, ltr), flat(Fte, Xte, lte)
    out = np.zeros((len(B), 5), dtype=np.float32)
    for s in range(5):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=s,
                              n_jobs=2, deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(A, Rtr[:, s])
        out[:, s] = m.predict(B)
    return out


if __name__ == "__main__":
    d = h.load()
    S = np.zeros_like(d["R"])
    for tr, va in h.folds(d["groups"]):
        S[va] = fit_predict_lgb(d["X"][tr], d["R"][tr], d["lead"][tr], d["X"][va], d["lead"][va])
    h.report("tdl__lgbref", S, d["R"], d["lead"])
    print("boundary_se", h.boundary_se(S, d["R"], d["groups"]))
