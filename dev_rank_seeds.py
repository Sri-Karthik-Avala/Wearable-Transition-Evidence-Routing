import numpy as np, lightgbm as lgb, wf_harness as h
import solution as S
d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); n = len(X); t1 = R.argmax(1)
_, St = S.site_stats(X)
rel = St / (np.abs(St).mean(1, keepdims=True) + 1e-6)
eye = np.eye(5, dtype=np.float32)
long = np.concatenate([St.reshape(n * 5, -1), rel.reshape(n * 5, -1), np.repeat(St.mean(1), 5, axis=0),
                       np.tile(eye, (n, 1)), np.repeat(lead[:, None], 5, axis=0)], 1).astype(np.float32)
label = (4 - np.argsort(np.argsort(-R, 1), 1)).reshape(-1).astype(int)
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)
sc = lambda S_, rows=None: float(h.row_scores(S_ if rows is None else S_[rows], R if rows is None else R[rows]).mean())
V3 = 0.35 * z(np.load('oof/v1__nn.npy')) + 0.45 * z(np.load('oof/an__base_xsite.npy')) + 0.2 * z(np.load('oof/v2__knn_p40k40.npy'))
LG = np.load('oof/an__base_xsite.npy')
seeds = (0, 1, 2)
Os = []
for sd in seeds:
    O = np.zeros_like(R)
    for a, b in folds:
        ai = (a[:, None] * 5 + np.arange(5)).ravel(); bi = (b[:, None] * 5 + np.arange(5)).ravel()
        m = lgb.LGBMRanker(objective='lambdarank', label_gain=[0.10, 0.25, 0.45, 0.70, 1.00], eval_at=[5],
                           n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.5, n_jobs=6, deterministic=True, force_row_wise=True,
                           verbose=-1, lambdarank_truncation_level=5, random_state=sd)
        m.fit(long[ai], label[ai], group=np.full(len(a), 5))
        O[b] = m.predict(long[bi]).reshape(-1, 5)
    Os.append(O)
    print(f'seed {sd}: ranker OOF {sc(O):.4f}  per-fold {[round(sc(O,b),4) for _,b in folds]}', flush=True)
    np.save(f'oof/rank__lmart_s{sd}.npy', O)
A = np.mean([z(o) for o in Os], 0)
np.save('oof/rank__lmart_avg3.npy', A)
print(f'\nseed-avg ranker OOF {sc(A):.4f}', flush=True)
print(f'LGB member per-fold {[round(sc(LG,b),4) for _,b in folds]}', flush=True)
print(f'v3 per-fold        {[round(sc(V3,b),4) for _,b in folds]}', flush=True)
print('\n=== seed-averaged ranker blended into v3 ===', flush=True)
for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    B = (1 - w) * V3 + w * A
    print(f'  w {w:.1f} OOF {sc(B):.4f} top1 {(B.argmax(1)==t1).mean():.4f} per-fold delta {[round(sc(B,b)-sc(V3,b),4) for _,b in folds]}', flush=True)
grid = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
out = np.zeros_like(R); picks = []
for a, b in folds:
    best = max(grid, key=lambda w: sc((1 - w) * V3 + w * A, a))
    picks.append(best); out[b] = ((1 - best) * V3 + best * A)[b]
print(f'\nNESTED (seed-avg): OOF {sc(out):.4f} picks {picks} top1 {(out.argmax(1)==t1).mean():.4f}  [v3 {sc(V3):.4f}]', flush=True)
