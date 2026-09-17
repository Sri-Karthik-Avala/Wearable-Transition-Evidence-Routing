import numpy as np, lightgbm as lgb, wf_harness as h
import solution as S
_s = open('dev_anat.py', encoding='utf-8').read()
_ns = {}
exec(_s[:_s.index("d = h.load()")], _ns)
anat_features, PRM, EPS, decode, score_order = (_ns[k] for k in ('anat_features','PRM','EPS','decode','score_order'))
d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); t1 = R.argmax(1)
base = np.concatenate([S.flat_features(X, lead), S.cross_site_features(X)], 1)
F = base
pick = 'oof/an__base_xsite.npy'
Eb = np.load(pick)
print('gain model', pick, round(score_order(decode(Eb), R), 4), 'top1acc', round(float((Eb.argmax(1) == t1).mean()), 4), flush=True)
P = np.zeros_like(R)
y = t1
for a, b in folds:
    m = lgb.LGBMClassifier(objective="multiclass", **PRM, random_state=0).fit(F[a], y[a])
    tmp = np.zeros((len(b), 5), np.float32)
    tmp[:, m.classes_] = m.predict_proba(F[b])
    P[b] = tmp
np.save('oof/an__top1_fixed.npy', P)
print('[B] top1 alone OOF', round(score_order(decode(P), R), 4), 'top1acc', round(float((P.argmax(1) == t1).mean()), 4), flush=True)
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + EPS)
best_w, best_sc = 0.0, score_order(decode(Eb), R)
for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    C = (1 - w) * z(Eb) + w * z(P)
    sc = score_order(decode(Eb, first=C.argmax(1)), R)
    print(f'[B] first={1-w:.1f}gain+{w:.1f}top1 OOF {sc:.4f} top1acc {(C.argmax(1)==t1).mean():.4f}', flush=True)
    if sc > best_sc: best_w, best_sc = w, sc
C = (1 - best_w) * z(Eb) + best_w * z(P) if best_w else z(Eb)
first = C.argmax(1)
order = decode(Eb, first=first)
print(f'[B] chosen w={best_w} OOF {best_sc:.4f}', flush=True)
for (i, j) in [(0, 1), (3, 4)]:
    yp = (R[:, i] > R[:, j]).astype(int)
    top2 = np.argsort(-R, 1)[:, :2]
    w_ = 1.0 + 2.0 * np.array([set([i, j]) == set(r) for r in top2])
    pp = np.zeros(len(F), np.float32)
    for a, b in folds:
        pp[b] = lgb.LGBMClassifier(**PRM, random_state=0).fit(F[a], yp[a], sample_weight=w_[a]).predict_proba(F[b])[:, 1]
    sel = np.array([set(o[:2]) == {i, j} for o in order])
    nf = first.copy(); nf[sel] = np.where(pp[sel] > 0.5, i, j)
    sc = score_order(decode(Eb, first=nf), R)
    print(f'[C] {h.SITES[i]}/{h.SITES[j]} rows {sel.sum()} top1acc {(first[sel]==t1[sel]).mean():.3f} -> {(nf[sel]==t1[sel]).mean():.3f} OOF {sc:.4f}', flush=True)
    if sc > best_sc: best_sc, first, order = sc, nf, decode(Eb, first=nf)
print(f'[FINAL] OOF {best_sc:.4f} top1acc {(first==t1).mean():.4f} (v3 ref 0.7689)', flush=True)
