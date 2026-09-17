"""KEY=err (k-series)  targeted ideas from the error analysis. Usage: dev_wf_err_k.py idea1,idea2,...
hurdle : E[g_s] = P(top=s) + (1-P(top=s)) * E[g_s | s not top]   (row multiclass on top site + conditional regressors)
top1mix: base regression + alpha * P(top=s)  (alpha picked on OOF of train folds only is NOT done; report a fixed grid)
"""
import sys, time, warnings
import numpy as np
import lightgbm as lgb
import wf_harness as h
warnings.filterwarnings("ignore")
t0 = time.time()
d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
n = len(X); FOLDS = h.folds(groups); GAINS = h.GAINS


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


P = dict(learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, n_jobs=2, deterministic=True, force_row_wise=True, verbose=-1)


def reg(A, y, B, seed, n_est=300):
    m = lgb.LGBMRegressor(n_estimators=n_est, random_state=seed, **P); m.fit(A, y); return m.predict(B)


def top_proba(A, top, B, seed, n_est=250):
    m = lgb.LGBMClassifier(n_estimators=n_est, random_state=seed, **P); m.fit(A, top)
    out = np.zeros((len(B), 5)); out[:, m.classes_] = m.predict_proba(B); return out


def fold_models(A, name_prefix):
    top = R.argmax(1)
    Sreg = np.zeros((n, 5)); Ptop = np.zeros((n, 5)); Scond = np.zeros((n, 5))
    for tr, va in FOLDS:
        Ptop[va] = top_proba(A[tr], top[tr], A[va], seed=0)
        for s in range(5):
            Sreg[va, s] = reg(A[tr], R[tr, s], A[va], seed=s)
            nt = tr[top[tr] != s]
            Scond[va, s] = reg(A[nt], R[nt, s], A[va], seed=s)
        print("   fold t", round(time.time() - t0, 1), flush=True)
    return Sreg, Ptop, Scond


if __name__ == "__main__":
    ideas = sys.argv[1].split(",")
    A = base_flat(X, lead)
    if "hurdle" in ideas or "top1mix" in ideas:
        Sreg, Ptop, Scond = fold_models(A, "err__k")
        h.report("err__k_reg", Sreg, R, lead)
        h.report("err__k_ptop", Ptop, R, lead)
        top = R.argmax(1)
        print("   ptop top-1 acc %.4f | reg top-1 acc %.4f" % ((Ptop.argmax(1) == top).mean(), (Sreg.argmax(1) == top).mean()))
        Sh = Ptop + (1 - Ptop) * Scond
        h.report("err__k_hurdle", Sh, R, lead)
        for a in (0.25, 0.5, 1.0):
            h.report(f"err__k_top1mix{a}", Sreg + a * Ptop, R, lead, save=False)
    print("done t", round(time.time() - t0, 1))
