import numpy as np, lightgbm as lgb, wf_harness as h
from sklearn.model_selection import GroupKFold
import solution as S
d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); t1 = R.argmax(1)
F = np.concatenate([S.flat_features(X, lead), S.cross_site_features(X)], 1).astype(np.float32)
del X
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)
NN = z(np.load('oof/v1__nn.npy')); KN = z(np.load('oof/v2__knn_p40k40.npy')); LG0 = np.load('oof/an__base_xsite.npy')
V3 = 0.35 * NN + 0.45 * z(LG0) + 0.2 * KN
sc = lambda M, rows=None: float(h.row_scores(M if rows is None else M[rows], R if rows is None else R[rows]).mean())
BASE = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.5, n_jobs=3, deterministic=True, force_row_wise=True, verbose=-1)
print('features', F.shape, '| v3', round(sc(V3), 4), '| lgb ref', round(sc(LG0), 4), flush=True)


def report(name, O):
    np.save(f'oof/lo__{name}.npy', O)
    swap = 0.35 * NN + 0.45 * z(O) + 0.2 * KN
    per = [round(sc(swap, b) - sc(V3, b), 4) for _, b in folds]
    print(f'[{name}] OOF {sc(O):.4f} top1 {(O.argmax(1)==t1).mean():.4f} | as v3 member {sc(swap):.4f} per-fold {per} ({sum(1 for x in per if x>0)}/5 up)', flush=True)


GRID = [dict(num_leaves=7, min_child_samples=10), dict(num_leaves=7, min_child_samples=40),
        dict(num_leaves=31, min_child_samples=20), dict(num_leaves=31, min_child_samples=40),
        dict(num_leaves=15, min_child_samples=20, learning_rate=0.015, n_estimators=600),
        dict(num_leaves=15, min_child_samples=20, colsample_bytree=0.3)]
O = np.zeros_like(R); picks = []
for a, b in folds:
    inner = list(GroupKFold(n_splits=2).split(np.zeros(len(a)), groups=g[a]))
    best, best_v = dict(), -1.0
    for cfg in GRID:
        prm = dict(BASE); prm.update(cfg)
        P = np.zeros((len(a), 5), np.float32)
        for ia, ib in inner:
            for s in range(5):
                P[ib, s] = lgb.LGBMRegressor(**prm, random_state=s).fit(F[a][ia], R[a][ia, s]).predict(F[a][ib])
        v = float(h.row_scores(P, R[a]).mean())
        if v > best_v: best, best_v = cfg, v
    picks.append(best)
    prm = dict(BASE); prm.update(best)
    for s in range(5):
        O[b, s] = lgb.LGBMRegressor(**prm, random_state=s).fit(F[a], R[a, s]).predict(F[b])
report('lgb_nested_hpo', O)
print('  chosen per fold:', picks, flush=True)

for K in (150, 300):
    O = np.zeros_like(R)
    for a, b in folds:
        imp = np.zeros(F.shape[1])
        for s in range(5):
            imp += lgb.LGBMRegressor(**BASE, random_state=s).fit(F[a], R[a, s]).feature_importances_
        keep = np.argsort(-imp)[:K]
        Fa, Fb = np.ascontiguousarray(F[a][:, keep]), np.ascontiguousarray(F[b][:, keep])
        for s in range(5):
            O[b, s] = lgb.LGBMRegressor(**BASE, random_state=s).fit(Fa, R[a, s]).predict(Fb)
    report(f'lgb_top{K}feats', O)
