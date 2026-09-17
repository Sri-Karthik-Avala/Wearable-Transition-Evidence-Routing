import numpy as np, lightgbm as lgb, wf_harness as h
import solution as S
PRM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.5, n_jobs=6, deterministic=True, force_row_wise=True, verbose=-1)
CH_ACC_Y, CH_MAG_Y = 1 * 9 + 1, 1 * 9 + 7


def feats(X, l):
    return np.concatenate([S.flat_features(X, l), S.cross_site_features(X)], 1)


def rand_rot(rng, k, max_deg):
    ax = rng.normal(size=(k, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    th = np.deg2rad(rng.uniform(-max_deg, max_deg, size=k))
    K = np.zeros((k, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    I = np.eye(3)[None]
    return I + np.sin(th)[:, None, None] * K + (1 - np.cos(th))[:, None, None] * (K @ K)


def rotate(X, seed, max_deg=25.0):
    rng = np.random.default_rng(seed)
    n = len(X); Z = X.reshape(n, 8, 5, 9).copy()
    for s in range(5):
        Rm = rand_rot(rng, n, max_deg)
        for k in (0, 3, 6):
            v = Z[:, :, s, k:k + 3]
            Z[:, :, s, k:k + 3] = np.einsum('nij,ntj->nti', Rm, v)
    return Z.reshape(n, 8, 45)


def run(name, Xtr_list, w_mode, Xva, lva, R, a, b, out):
    Xa = np.concatenate(Xtr_list, 0)
    la = np.tile(lead[a], len(Xtr_list))
    Ra = np.tile(R[a], (len(Xtr_list), 1))
    A, B = feats(Xa, la), feats(Xva, lva)
    for s in range(5):
        w = None
        if w_mode == 'gain':
            w = 1.0 + 2.0 * (Ra[:, s] >= 0.70)
        out[b, s] = lgb.LGBMRegressor(**PRM, random_state=s).fit(A, Ra[:, s], sample_weight=w).predict(B)


d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); t1 = R.argmax(1)
CONF = {'baseline': (1, None), 'rot_aug_x3': (3, None), 'gain_weighted': (1, 'gain'), 'rot_aug_x3_gw': (3, 'gain')}
res = {}
for name, (ncopy, wm) in CONF.items():
    O = np.zeros_like(R); Ostress = np.zeros_like(R)
    for a, b in folds:
        parts = [X[a]] + [rotate(X[a], 100 + i) for i in range(ncopy - 1)]
        run(name, parts, wm, X[b], lead[b], R, a, b, O)
        Xs = X[b].copy(); Xs[:, :, CH_ACC_Y] *= 7.6; Xs[:, :, CH_MAG_Y] *= 3.8
        run(name, parts, wm, Xs, lead[b], R, a, b, Ostress)
    res[name] = O
    np.save(f'oof/aug__{name}.npy', O)
    print(f'{name:16s} OOF {h.row_scores(O,R).mean():.4f} top1 {(O.argmax(1)==t1).mean():.4f} | under RUpArm stress {h.row_scores(Ostress,R).mean():.4f}', flush=True)
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)
V3 = 0.35 * z(np.load('oof/v1__nn.npy')) + 0.45 * z(np.load('oof/an__base_xsite.npy')) + 0.2 * z(np.load('oof/v2__knn_p40k40.npy'))
print(f'\nv3 ref {h.row_scores(V3,R).mean():.4f}', flush=True)
for name, O in res.items():
    if name == 'baseline':
        continue
    swap = 0.35 * z(np.load('oof/v1__nn.npy')) + 0.45 * z(O) + 0.2 * z(np.load('oof/v2__knn_p40k40.npy'))
    per = [round(float(h.row_scores(swap[b], R[b]).mean() - h.row_scores(V3[b], R[b]).mean()), 4) for _, b in folds]
    print(f'  v3 with LGB member = {name}: {h.row_scores(swap,R).mean():.4f} per-fold delta {per}', flush=True)
