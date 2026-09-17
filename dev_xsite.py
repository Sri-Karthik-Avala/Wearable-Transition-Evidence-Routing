import numpy as np, lightgbm as lgb, wf_harness as h
import solution as S
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def base_feats(X, l):
    _, Fs = S.site_stats(X)
    n = len(Fs)
    rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
    return np.concatenate([Fs.reshape(n, -1), rel.reshape(n, -1), X[:, -1], l[:, None]], 1).astype(np.float32)


def xsite_feats(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9)
    out = []
    for k in (0, 3, 6):
        v = Xs[..., k:k + 3]
        nrm = np.linalg.norm(v, axis=-1)
        nc = nrm - nrm.mean(1, keepdims=True)
        mv = v.mean(1)
        mvn = mv / (np.linalg.norm(mv, axis=-1, keepdims=True) + 1e-6)
        dv = v[:, -1] - v[:, 0]
        for i, j in PAIRS:
            cov = (nc[:, :, i] * nc[:, :, j]).mean(1)
            out.append(cov / (nc[:, :, i].std(1) * nc[:, :, j].std(1) + 1e-6))
            out.append((mvn[:, i] * mvn[:, j]).sum(-1))
            out.append(np.linalg.norm(dv[:, i], axis=-1) - np.linalg.norm(dv[:, j], axis=-1))
    sd = Xs.std(1).mean(-1)
    out.append(np.argsort(np.argsort(sd, 1), 1).astype(np.float32).T)
    return np.column_stack([o if o.ndim == 1 else o.T for o in out]).astype(np.float32)


def run(F, R, groups, lead, name, seeds=(0,)):
    O = np.zeros_like(R)
    for a, b in h.folds(groups):
        for s in range(5):
            for sd in seeds:
                m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=s + 7 * sd,
                                      n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
                m.fit(F[a], R[a, s]); O[b, s] += m.predict(F[b]) / len(seeds)
    return h.report(name, O, R, lead)


d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
B = base_feats(X, lead); XS = xsite_feats(X)
print('feat dims', B.shape, XS.shape, flush=True)
run(B, R, g, lead, 'xs__base')
run(np.concatenate([B, XS], 1), R, g, lead, 'xs__base_xsite')
