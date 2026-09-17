import numpy as np, lightgbm as lgb, wf_harness as h
from sklearn.isotonic import IsotonicRegression
import solution as S
d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); n = len(X); t1 = R.argmax(1)
_, St = S.site_stats(X)
mean_stats = St.mean(1)
rel = St / (np.abs(St).mean(1, keepdims=True) + 1e-6)
eye = np.eye(5, dtype=np.float32)
long = np.concatenate([St.reshape(n * 5, -1), rel.reshape(n * 5, -1),
                       np.repeat(mean_stats, 5, axis=0), np.tile(eye, (n, 1)),
                       np.repeat(lead[:, None], 5, axis=0)], 1).astype(np.float32)
rank_pos = np.argsort(np.argsort(-R, 1), 1)
label = (4 - rank_pos).reshape(-1).astype(int)
GAINS_ASC = [0.10, 0.25, 0.45, 0.70, 1.00]
print('long format', long.shape, flush=True)
O = np.zeros_like(R)
for a, b in folds:
    ai = (a[:, None] * 5 + np.arange(5)).ravel(); bi = (b[:, None] * 5 + np.arange(5)).ravel()
    m = lgb.LGBMRanker(objective='lambdarank', label_gain=GAINS_ASC, eval_at=[5], n_estimators=400,
                       learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.5, n_jobs=6, deterministic=True, force_row_wise=True, verbose=-1,
                       lambdarank_truncation_level=5, random_state=0)
    m.fit(long[ai], label[ai], group=np.full(len(a), 5))
    O[b] = m.predict(long[bi]).reshape(-1, 5)
np.save('oof/rank__lambdamart_metricgains.npy', O)
print(f'[A] metric-aligned LambdaMART OOF {h.row_scores(O,R).mean():.4f} top1acc {(O.argmax(1)==t1).mean():.4f}  (v3 LGB member 0.7640)', flush=True)
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)
V3 = 0.35 * z(np.load('oof/v1__nn.npy')) + 0.45 * z(np.load('oof/an__base_xsite.npy')) + 0.2 * z(np.load('oof/v2__knn_p40k40.npy'))
print(f'    v3 ref {h.row_scores(V3,R).mean():.4f}', flush=True)
for w in (0.2, 0.3, 0.4):
    print(f'    v3 + {w}*lambdamart -> {h.row_scores((1-w)*V3+w*z(O),R).mean():.4f}', flush=True)
print(flush=True)
for nm in ['an__base_xsite', 'v1__nn']:
    P = np.load(f'oof/{nm}.npy')
    C = np.zeros_like(P)
    for k, (a, b) in enumerate(folds):
        for s in range(5):
            iso = IsotonicRegression(out_of_bounds='clip').fit(P[a, s], R[a, s])
            C[b, s] = iso.predict(P[b, s])
    print(f'[B] {nm}: raw {h.row_scores(P,R).mean():.4f} -> per-site isotonic {h.row_scores(C,R).mean():.4f} '
          f'(top1 {(P.argmax(1)==t1).mean():.4f} -> {(C.argmax(1)==t1).mean():.4f})', flush=True)
    np.save(f'oof/cal__{nm}.npy', C)
Pl = np.load('oof/an__base_xsite.npy'); Pn = np.load('oof/v1__nn.npy')
Cl = np.load('oof/cal__an__base_xsite.npy'); Cn = np.load('oof/cal__v1__nn.npy')
kn = np.load('oof/v2__knn_p40k40.npy')
B2 = 0.35 * z(Cn) + 0.45 * z(Cl) + 0.2 * z(kn)
print(f'[B] v3 with both members calibrated: {h.row_scores(B2,R).mean():.4f} (v3 {h.row_scores(V3,R).mean():.4f})', flush=True)
