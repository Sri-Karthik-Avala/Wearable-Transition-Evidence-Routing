import numpy as np, lightgbm as lgb, wf_harness as h
import solution as S
PRM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.5, n_jobs=6, deterministic=True, force_row_wise=True, verbose=-1)
CH_ACC_Y, CH_MAG_Y = 1 * 9 + 1, 1 * 9 + 7
CLIP_Q = 0.995


def feats(X, l):
    return np.concatenate([S.flat_features(X, l), S.cross_site_features(X)], 1)


def stress(X, k_acc, k_mag):
    Z = X.copy()
    Z[:, :, CH_ACC_Y] *= k_acc
    Z[:, :, CH_MAG_Y] *= k_mag
    return Z


def clip_to(X, lo, hi):
    return np.clip(X, lo[None, None, :], hi[None, None, :])


d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g)
variants = {'clean': (1.0, 1.0), 'stress_test_like': (7.6, 3.8), 'stress_mild': (3.0, 2.0)}
O = {('plain', v): np.zeros_like(R) for v in variants}
O.update({('clipped', v): np.zeros_like(R) for v in variants})
for a, b in folds:
    lo = np.percentile(X[a].reshape(-1, 45), 100 * (1 - CLIP_Q), axis=0)
    hi = np.percentile(X[a].reshape(-1, 45), 100 * CLIP_Q, axis=0)
    Fa_plain = feats(X[a], lead[a])
    Fa_clip = feats(clip_to(X[a], lo, hi), lead[a])
    models = {}
    for tag, Fa in [('plain', Fa_plain), ('clipped', Fa_clip)]:
        models[tag] = [lgb.LGBMRegressor(**PRM, random_state=s).fit(Fa, R[a, s]) for s in range(5)]
    for vname, (ka, km) in variants.items():
        Xv = stress(X[b], ka, km)
        for tag in ('plain', 'clipped'):
            Fv = feats(Xv if tag == 'plain' else clip_to(Xv, lo, hi), lead[b])
            for s in range(5):
                O[(tag, vname)][b, s] = models[tag][s].predict(Fv)
for tag in ('plain', 'clipped'):
    for vname in variants:
        sc = h.row_scores(O[(tag, vname)], R).mean()
        print(f'{tag:8s} {vname:17s} OOF {sc:.4f} top1acc {(O[(tag,vname)].argmax(1)==R.argmax(1)).mean():.4f}', flush=True)
    np.save(f'oof/stress__{tag}_clean.npy', O[(tag, 'clean')])
