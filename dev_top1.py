import numpy as np, lightgbm as lgb, wf_harness as h
import solution as S
d = h.load(); R, X, lead = d['R'], d['X'], d['lead']
F5 = h.folds(d['groups'])
_, Fs = S.site_stats(X)
n = len(X)
rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
A = np.concatenate([Fs.reshape(n, -1), rel.reshape(n, -1), X[:, -1], lead[:, None]], 1)
t1 = R.argmax(1)
P = np.zeros((n, 5), np.float32)
for a, b in F5:
    m = lgb.LGBMClassifier(objective='multiclass', n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=4, verbose=-1, random_state=0)
    m.fit(A[a], t1[a]); P[b] = m.predict_proba(A[b])
h.report('top1__lgb_multiclass', P, R, lead)
reg = np.load('oof/tgt__ref_siteReg.npy')
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)
for w in (0.2, 0.35, 0.5):
    print('reg+top1', w, round(float(h.row_scores((1 - w) * z(reg) + w * z(P), R).mean()), 4))
knn = np.load('oof/tgt__knn_k25_pca32_expgain.npy')
for w in (0.2, 0.35):
    print('reg+top1+knn', w, round(float(h.row_scores(z(reg) + w * z(P) + 0.5 * z(knn), R).mean()), 4))
print('top1 acc multiclass', (P.argmax(1) == t1).mean().round(3), 'reg', (reg.argmax(1) == t1).mean().round(3))
