import numpy as np, lightgbm as lgb, wf_harness as h
import solution as S
EPS = 1e-6
PRM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.5, n_jobs=6, deterministic=True, force_row_wise=True, verbose=-1)
d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); t1 = R.argmax(1)
F = np.concatenate([S.flat_features(X, lead), S.cross_site_features(X)], 1)
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + EPS)
nn = np.load('oof/v1__nn.npy'); xs = np.load('oof/an__base_xsite.npy'); kn = np.load('oof/v2__knn_p40k40.npy')
B = 0.35 * z(nn) + 0.45 * z(xs) + 0.2 * z(kn)
sc = lambda S_: round(float(h.row_scores(S_, R).mean()), 4)
print('v3 blend', sc(B), 'top1acc', round(float((B.argmax(1) == t1).mean()), 4), flush=True)
PP = {}
for (i, j) in [(0, 1), (3, 4)]:
    y = (R[:, i] > R[:, j]).astype(int)
    top2 = np.argsort(-R, 1)[:, :2]
    w = 1.0 + 2.0 * np.array([set([i, j]) == set(r) for r in top2])
    p = np.zeros(len(F), np.float32)
    for a, b in folds:
        p[b] = lgb.LGBMClassifier(**PRM, random_state=0).fit(F[a], y[a], sample_weight=w[a]).predict_proba(F[b])[:, 1]
    PP[(i, j)] = p
    np.save(f'oof/pair__{i}{j}.npy', p)


def apply_pair(Sc, i, j, thr=None):
    order = np.argsort(-Sc, 1, kind="stable")
    sel = np.array([set(o[:2]) == {i, j} for o in order])
    if thr is not None:
        srt = np.sort(Sc, 1)[:, ::-1]
        sel &= (srt[:, 0] - srt[:, 1]) < thr
    out = Sc.copy()
    idx = np.where(sel)[0]
    win = np.where(PP[(i, j)][idx] > 0.5, i, j)
    lose = np.where(win == i, j, i)
    big = Sc[idx].max(1) + 1.0
    out[idx, win] = big
    out[idx, lose] = big - 0.5
    return out, sel.sum()


for (i, j) in [(3, 4), (0, 1)]:
    Sn, k = apply_pair(B, i, j)
    print(f'blend + pair {h.SITES[i]}/{h.SITES[j]} (all top-2 rows {k}) OOF {sc(Sn)}', flush=True)
Sleg, _ = apply_pair(B, 3, 4)
print('blend + leg only', sc(Sleg), 'top1acc', round(float((Sleg.argmax(1) == t1).mean()), 4), flush=True)
srt = np.sort(B, 1)[:, ::-1]; marg = srt[:, 0] - srt[:, 1]
for thr in (0.2, 0.4, 0.6, 1.0, 2.0):
    Sm, k = apply_pair(B, 3, 4, thr=thr)
    print(f'leg reranker margin<{thr} rows {k} OOF {sc(Sm)}', flush=True)
nested = []
for a, b in folds:
    best_thr, best_v = None, -1
    for thr in (0.0, 0.2, 0.4, 0.6, 1.0, 2.0, 9.0):
        Sm, _ = apply_pair(B, 3, 4, thr=thr if thr < 9 else None)
        v = h.row_scores(Sm[a], R[a]).mean()
        if v > best_v: best_thr, best_v = thr, v
    Sm, _ = apply_pair(B, 3, 4, thr=best_thr if best_thr < 9 else None)
    nested.append((b, Sm[b], best_thr))
O = B.copy()
for b, sm, _ in nested: O[b] = sm
print('NESTED leg reranker (threshold chosen per fold)', sc(O), 'thresholds', [t for _, _, t in nested], flush=True)
