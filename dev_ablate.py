import numpy as np, lightgbm as lgb, wf_harness as h
EPS = 1e-6
PRM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.5, n_jobs=6, deterministic=True, force_row_wise=True, verbose=-1)
TYPE_SLICES = {'acc': slice(0, 3), 'gyro': slice(3, 6), 'mag': slice(6, 9)}


def build(X, lead, types, mode='raw'):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9)
    if mode == 'center':
        Xs = Xs - Xs.mean(1, keepdims=True)
    elif mode == 'tz':
        Xs = (Xs - Xs.mean(1, keepdims=True)) / (Xs.std(1, keepdims=True) + EPS)
    elif mode == 'magcenter':
        Xs = Xs.copy(); Xs[..., 6:9] -= Xs[..., 6:9].mean(1, keepdims=True)
    parts = []
    for t in types:
        v = Xs[..., TYPE_SLICES[t]]
        nr = np.linalg.norm(v, axis=-1)
        dv = np.abs(np.diff(v, axis=1)).mean(1)
        parts += [v.std(1), v.mean(1), v[:, -1] - v[:, 0], dv, v[:, -2:].mean(1) - v[:, :-2].mean(1),
                  nr.std(1)[..., None], nr.mean(1)[..., None], np.abs(np.diff(nr, axis=1)).mean(1)[..., None]]
    P = np.concatenate(parts, -1)
    rel = P / (np.abs(P).mean(1, keepdims=True) + EPS)
    return np.concatenate([P.reshape(n, -1), rel.reshape(n, -1), lead[:, None]], 1).astype(np.float32)


def oof(F, R, folds):
    O = np.zeros_like(R)
    for a, b in folds:
        for s in range(5):
            O[b, s] = lgb.LGBMRegressor(**PRM, random_state=s).fit(F[a], R[a, s]).predict(F[b])
    return O


d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); t1 = R.argmax(1)
CONFIGS = [('ALL', ['acc', 'gyro', 'mag'], 'raw'), ('ACC', ['acc'], 'raw'), ('GYRO', ['gyro'], 'raw'), ('MAG', ['mag'], 'raw'),
           ('ACC+GYRO', ['acc', 'gyro'], 'raw'), ('ACC+MAG', ['acc', 'mag'], 'raw'), ('GYRO+MAG', ['gyro', 'mag'], 'raw'),
           ('ALL_center', ['acc', 'gyro', 'mag'], 'center'), ('ALL_tzscore', ['acc', 'gyro', 'mag'], 'tz'),
           ('ALL_magcenter', ['acc', 'gyro', 'mag'], 'magcenter')]
for nm, types, mode in CONFIGS:
    F = build(X, lead, types, mode)
    O = oof(F, R, folds)
    np.save(f'oof/abl__{nm}.npy', O)
    print(f'{nm:14s} dims {F.shape[1]:4d} OOF {h.row_scores(O, R).mean():.4f} top1acc {(O.argmax(1)==t1).mean():.4f}', flush=True)
