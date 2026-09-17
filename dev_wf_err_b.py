"""KEY=err  targeted feature/model ideas from the error analysis. Usage: dev_wf_err_b.py idea1,idea2,..."""
import sys, time, warnings
import numpy as np
import lightgbm as lgb
import wf_harness as h
warnings.filterwarnings("ignore")
t0 = time.time()
d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
n = len(X); FOLDS = h.folds(groups)
GAINS = h.GAINS


def site_stats(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)
    norms = np.stack([np.linalg.norm(Xs[..., k:k + 3], axis=-1) for k in (0, 3, 6)], -1)
    parts = [Xs.std(2), Xs.mean(2), Xs[:, :, -1] - Xs[:, :, 0], np.abs(np.diff(Xs, axis=2)).mean(2),
             Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2), norms.std(2), norms.mean(2),
             np.abs(np.diff(norms, axis=2)).mean(2)]
    return np.concatenate(parts, -1).astype(np.float32)


def base_flat(X, l):
    Fs = site_stats(X); n = len(Fs)
    rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
    return np.concatenate([Fs.reshape(n, -1), rel.reshape(n, -1), X[:, -1], l[:, None]], 1)


def rank_in_row(v):  # v n,5,... -> rank across sites (axis 1)
    return np.argsort(np.argsort(v, 1), 1).astype(np.float32)


def type_feats(X):
    """Per site x channel-type (acc/gyro/mag) summaries mimicking the target's |change|/(pre_std) structure."""
    n = len(X)
    Xs = X.reshape(n, 8, 5, 3, 3)  # n,t,site,type,axis
    pre = Xs[:, :-1]
    sd = Xs.std(1); sdpre = pre.std(1)
    z = np.abs(Xs[:, -1] - pre.mean(1)) / (sdpre + 0.05)
    z2 = np.abs(Xs[:, -2:].mean(1) - Xs[:, :-2].mean(1)) / (Xs[:, :-2].std(1) + 0.05)
    tt = np.arange(8, dtype=np.float32) - 3.5
    slope = np.abs((Xs * tt[None, :, None, None, None]).sum(1) / (tt ** 2).sum()) / (sd + 0.05)
    ld = np.abs(Xs[:, -1] - Xs[:, -2])
    ad = np.abs(np.diff(Xs, axis=1)).mean(1)
    lr = np.log((Xs[:, 4:].std(1) + 1e-2) / (Xs[:, :4].std(1) + 1e-2))
    feats = [np.log(sd + 1e-3), z, z2, slope, np.log(ld + 1e-3), np.log(ad + 1e-3), lr]
    per = np.stack([f.mean(-1) for f in feats], -1)  # n,5,3,K  (axis-mean per type)
    rk = rank_in_row(per)  # rank across sites of each (type,stat)
    rel = per - per.mean(1, keepdims=True)
    return per.reshape(n, -1), rk.reshape(n, -1), rel.reshape(n, -1)


def lgb_reg(A, y, B, seed=0, n_est=300):
    m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, random_state=seed, n_jobs=2, deterministic=True,
                          force_row_wise=True, verbose=-1)
    m.fit(A, y); return m.predict(B)


def run_wide(A, name):
    S = np.zeros((n, 5), np.float32)
    for tr, va in FOLDS:
        for s in range(5):
            S[va, s] = lgb_reg(A[tr], R[tr, s], A[va], seed=s)
    h.report(name, S, R, lead); print("   t", round(time.time() - t0, 1), flush=True)
    return S


def run_cls(A, name):
    """Per-site multiclass on true rank position -> expected gain (Bayes decode)."""
    pos = np.argsort(np.argsort(-R, 1), 1)  # 0 = top
    S = np.zeros((n, 5), np.float32)
    for tr, va in FOLDS:
        for s in range(5):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=s, n_jobs=2,
                                   deterministic=True, force_row_wise=True, verbose=-1)
            m.fit(A[tr], pos[tr, s]); P = m.predict_proba(A[va])
            S[va, s] = P @ GAINS[m.classes_]
    h.report(name, S, R, lead); print("   t", round(time.time() - t0, 1), flush=True)
    return S


def run_long(Lf, name):
    """Site-stacked model: one regressor over all 5 sites (site id categorical), row-context features."""
    S = np.zeros((n, 5), np.float32)
    for tr, va in FOLDS:
        A = Lf[tr].reshape(-1, Lf.shape[-1]); B = Lf[va].reshape(-1, Lf.shape[-1])
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, random_state=0, n_jobs=2, deterministic=True,
                              force_row_wise=True, verbose=-1)
        m.fit(A, R[tr].reshape(-1), categorical_feature=[0]); S[va] = m.predict(B).reshape(len(va), 5)
    h.report(name, S, R, lead); print("   t", round(time.time() - t0, 1), flush=True)
    return S


if __name__ == "__main__":
    ideas = sys.argv[1].split(",")
    base = base_flat(X, lead)
    per, rk, rel = type_feats(X)
    for idea in ideas:
        if idea == "type":  # baseline + type-separated evidence-proxy features + in-row ranks
            run_wide(np.concatenate([base, per, rk, rel], 1), "err__type")
        elif idea == "typeonly":
            run_wide(np.concatenate([per, rk, rel, lead[:, None]], 1), "err__typeonly")
        elif idea == "cls":
            run_cls(np.concatenate([base, per, rk, rel], 1), "err__type_cls")
        elif idea == "long":
            K = per.shape[1] // 5
            p5, r5, e5 = per.reshape(n, 5, K), rk.reshape(n, 5, K), rel.reshape(n, 5, K)
            sid = np.broadcast_to(np.arange(5, dtype=np.float32)[None, :, None], (n, 5, 1))
            ctx = np.broadcast_to(per[:, None, :], (n, 5, per.shape[1]))
            Lf = np.concatenate([sid, np.broadcast_to(lead[:, None, None], (n, 5, 1)), p5, r5, e5, ctx], -1)
            run_long(Lf, "err__type_long")
    print("done t", round(time.time() - t0, 1))
