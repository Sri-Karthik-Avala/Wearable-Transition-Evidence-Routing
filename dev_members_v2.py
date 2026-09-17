import numpy as np, lightgbm as lgb, wf_harness as h
from sklearn.decomposition import PCA
import solution as S


def flat_features(X, l):
    _, Fs = S.site_stats(X)
    n = len(Fs)
    rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
    return np.concatenate([Fs.reshape(n, -1), rel.reshape(n, -1), X[:, -1], l[:, None]], 1).astype(np.float32)


def fit_predict_top1(Xtr, Rtr, ltr, Xte, lte):
    A, B = flat_features(Xtr, ltr), flat_features(Xte, lte)
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.03, num_leaves=15,
                           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           random_state=0, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
    m.fit(A, Rtr.argmax(1))
    out = np.zeros((len(B), 5), dtype=np.float32)
    out[:, m.classes_] = m.predict_proba(B)
    return out


def fit_predict_knn(Xtr, Rtr, ltr, Xte, lte, n_pca=32, k=25):
    A, B = flat_features(Xtr, ltr), flat_features(Xte, lte)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    pca = PCA(n_pca, random_state=0).fit((A - mu) / sd)
    Za, Zb = pca.transform((A - mu) / sd), pca.transform((B - mu) / sd)
    out = np.zeros((len(B), 5), dtype=np.float32)
    for i in range(0, len(Zb), 256):
        d2 = ((Zb[i:i + 256, None, :] - Za[None]) ** 2).sum(-1)
        nn_idx = np.argsort(d2, 1, kind="stable")[:, :k]
        out[i:i + 256] = Rtr[nn_idx].mean(1)
    return out


if __name__ == "__main__":
    d = h.load(); R, X, lead = d["R"], d["X"], d["lead"]
    for name, fn in [("v2__top1", fit_predict_top1), ("v2__knn_p32k25", fit_predict_knn),
                     ("v2__knn_p40k40", lambda *a: fit_predict_knn(*a, n_pca=40, k=40))]:
        O = np.zeros_like(R)
        for a, b in h.folds(d["groups"]):
            O[b] = fn(X[a], R[a], lead[a], X[b], lead[b])
        h.report(name, O, R, lead)
