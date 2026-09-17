import numpy as np, lightgbm as lgb, wf_harness as h
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import solution as S
d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); t1 = R.argmax(1)
F = np.concatenate([S.flat_features(X, lead), S.cross_site_features(X)], 1).astype(np.float32)
del X
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)
nn = z(np.load('oof/v1__nn.npy')); lg = z(np.load('oof/an__base_xsite.npy')); kn = z(np.load('oof/v2__knn_p40k40.npy')); et0 = z(np.load('oof/lo__extratrees.npy'))
V4 = 0.3 * nn + 0.2 * lg + 0.1 * kn + 0.4 * et0
sc = lambda M, rows=None: float(h.row_scores(M if rows is None else M[rows], R if rows is None else R[rows]).mean())
print('features', F.shape, '| v4 ref', round(sc(V4), 4), '| ET ref', round(sc(et0), 4), flush=True)


def evaluate(name, make):
    O = np.zeros_like(R)
    for a, b in folds:
        m = make(); m.fit(F[a], R[a]); O[b] = m.predict(F[b])
    np.save(f'oof/tr__{name}.npy', O)
    swap = 0.3 * nn + 0.2 * lg + 0.1 * kn + 0.4 * z(O)
    per = [round(sc(swap, b) - sc(V4, b), 4) for _, b in folds]
    print(f'[{name}] OOF {sc(O):.4f} top1 {(O.argmax(1)==t1).mean():.4f} | in v4 slot {sc(swap):.4f} per-fold {per} ({sum(1 for x in per if x>0)}/5 up)', flush=True)
    return O


CONF = [
    ('et_600_l1_mf02', lambda: ExtraTreesRegressor(n_estimators=600, min_samples_leaf=1, max_features=0.2, n_jobs=3, random_state=0)),
    ('et_600_l2_mf01', lambda: ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.1, n_jobs=3, random_state=0)),
    ('et_600_l2_mf04', lambda: ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.4, n_jobs=3, random_state=0)),
    ('et_600_l5_mf02', lambda: ExtraTreesRegressor(n_estimators=600, min_samples_leaf=5, max_features=0.2, n_jobs=3, random_state=0)),
    ('rf_400_l2_mf03', lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.3, n_jobs=3, random_state=0)),
]
outs = {}
for nm, mk in CONF:
    outs[nm] = evaluate(nm, mk)

O = np.zeros_like(R)
for a, b in folds:
    for s in range(5):
        O[b, s] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, extra_trees=True,
                                    n_jobs=3, deterministic=True, force_row_wise=True, verbose=-1,
                                    random_state=s).fit(F[a], R[a, s]).predict(F[b])
np.save('oof/tr__lgb_extratrees_mode.npy', O)
outs['lgb_extra'] = O
swap = 0.3 * nn + 0.2 * lg + 0.1 * kn + 0.4 * z(O)
print(f'[lgb_extra_trees_mode] OOF {sc(O):.4f} top1 {(O.argmax(1)==t1).mean():.4f} | in v4 slot {sc(swap):.4f}', flush=True)

best = max(outs, key=lambda k: sc(outs[k]))
print(f'\nbest randomized-tree member: {best} ({sc(outs[best]):.4f})', flush=True)
B = z(outs[best])
print('=== ET-heavy blends with the best member ===', flush=True)
for w in (0.4, 0.5, 0.6, 0.7):
    r = 1 - w
    Sb = r * (0.5 * nn + 0.33 * lg + 0.17 * kn) + w * B
    per = [round(sc(Sb, b) - sc(V4, b), 4) for _, b in folds]
    print(f'  w {w:.1f}: OOF {sc(Sb):.4f} top1 {(Sb.argmax(1)==t1).mean():.4f} per-fold {per} ({sum(1 for x in per if x>0)}/5 up)', flush=True)
print('=== two randomized members together (best + original ET) ===', flush=True)
for w1, w2 in [(0.3, 0.3), (0.25, 0.25), (0.35, 0.25)]:
    Sb = (1 - w1 - w2) * (0.5 * nn + 0.33 * lg + 0.17 * kn) + w1 * B + w2 * et0
    per = [round(sc(Sb, b) - sc(V4, b), 4) for _, b in folds]
    print(f'  best {w1} + et0 {w2}: OOF {sc(Sb):.4f} per-fold {per} ({sum(1 for x in per if x>0)}/5 up)', flush=True)
