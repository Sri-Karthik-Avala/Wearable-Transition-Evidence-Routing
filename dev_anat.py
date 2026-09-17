import numpy as np, lightgbm as lgb, wf_harness as h
import solution as S
EPS = 1e-6
TYPES = (0, 3, 6)
REL_PAIRS = [(0, 1), (3, 4), (2, 0), (2, 3)]
NJ = 6
PRM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.5, n_jobs=NJ, deterministic=True, force_row_wise=True, verbose=-1)


def _slope(y):
    t = np.arange(y.shape[1], dtype=np.float32)
    tc = t - t.mean()
    sl = (y * tc).sum(1) / (tc ** 2).sum()
    fit = sl[:, None] * tc + y.mean(1)[:, None]
    return sl, (y - fit).std(1)


def anat_features(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9)
    out = []
    nrm_by_type = {}
    for k in TYPES:
        v = Xs[..., k:k + 3]
        nr = np.linalg.norm(v, axis=-1)
        nrm_by_type[k] = nr
        dv = np.diff(v, axis=1)
        dvn = np.linalg.norm(dv, axis=-1)
        c = (v[:, 1:] * v[:, :-1]).sum(-1) / (np.linalg.norm(v[:, 1:], axis=-1) * np.linalg.norm(v[:, :-1], axis=-1) + EPS)
        for s in range(5):
            sl, rs = _slope(nr[:, :, s])
            out += [nr[:, :, s].mean(1), nr[:, :, s].std(1), nr[:, -1, s] - nr[:, 0, s],
                    np.abs(np.diff(nr[:, :, s], axis=1)).mean(1), (dvn[:, :, s] ** 2).sum(1),
                    dvn[:, :, s].max(1) / (dvn[:, :, s].mean(1) + EPS), sl, rs, c[:, :, s].mean(1), c[:, :, s].min(1)]
    for k in TYPES:
        v = Xs[..., k:k + 3]
        mv = v.mean(1)
        nr = nrm_by_type[k]
        for i, j in REL_PAIRS:
            d = v[:, :, i] - v[:, :, j]
            dn = np.linalg.norm(d, axis=-1)
            dd = np.linalg.norm(np.diff(d, axis=1), axis=-1)
            sl, rs = _slope(dn)
            ci = nr[:, :, i] - nr[:, :, i].mean(1, keepdims=True)
            cj = nr[:, :, j] - nr[:, :, j].mean(1, keepdims=True)
            out += [dn.mean(1), dn.std(1), dn[:, -1] - dn[:, 0], np.abs(np.diff(dn, axis=1)).mean(1), (dd ** 2).sum(1),
                    (mv[:, i] * mv[:, j]).sum(-1) / (np.linalg.norm(mv[:, i], axis=-1) * np.linalg.norm(mv[:, j], axis=-1) + EPS),
                    (ci * cj).mean(1) / (ci.std(1) * cj.std(1) + EPS), sl]
    return np.column_stack(out).astype(np.float32)


def oof_reg(F, R, folds):
    O = np.zeros_like(R)
    for a, b in folds:
        for s in range(5):
            O[b, s] = lgb.LGBMRegressor(**PRM, random_state=s).fit(F[a], R[a, s]).predict(F[b])
    return O


def oof_top1(F, R, folds):
    P = np.zeros_like(R)
    y = R.argmax(1)
    for a, b in folds:
        m = lgb.LGBMClassifier(objective="multiclass", **PRM, random_state=0).fit(F[a], y[a])
        P[b][:, m.classes_] = m.predict_proba(F[b])
    return P


def oof_pair(F, R, folds, i, j):
    y = (R[:, i] > R[:, j]).astype(int)
    top2 = (np.argsort(-R, 1)[:, :2])
    both = np.array([set([i, j]) == set(r) for r in top2])
    w = 1.0 + 2.0 * both
    P = np.zeros(len(F), np.float32)
    for a, b in folds:
        m = lgb.LGBMClassifier(**PRM, random_state=0).fit(F[a], y[a], sample_weight=w[a])
        P[b] = m.predict_proba(F[b])[:, 1]
    return P


def decode(E, first=None):
    order = np.argsort(-E, 1, kind="stable")
    if first is None:
        return order
    out = np.empty_like(order)
    out[:, 0] = first
    for r in range(len(order)):
        rest = [c for c in order[r] if c != first[r]]
        out[r, 1:] = rest
    return out


def score_order(order, R):
    return float((((np.take_along_axis(R, order, 1) * h.DISC).sum(1)) - h.WORST) / (h.IDEAL - h.WORST)).__float__() if order.ndim == 1 else float(((((np.take_along_axis(R, order, 1) * h.DISC).sum(1)) - h.WORST) / (h.IDEAL - h.WORST)).mean())


d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g)
base = np.concatenate([S.flat_features(X, lead), S.cross_site_features(X)], 1)
anat = anat_features(X)
print('dims base', base.shape, 'anat', anat.shape, flush=True)
t1 = R.argmax(1)
SETS = {'base_xsite': base, 'base_xsite_anat': np.concatenate([base, anat], 1)}
E = {}
for nm, F in SETS.items():
    E[nm] = oof_reg(F, R, folds)
    print(f'[A] {nm} OOF {score_order(decode(E[nm]), R):.4f} top1acc {(E[nm].argmax(1)==t1).mean():.4f}', flush=True)
    np.save(f'oof/an__{nm}.npy', E[nm])
best = max(E, key=lambda k: score_order(decode(E[k]), R))
F = SETS[best]; Eb = E[best]
print('best set', best, flush=True)
P = oof_top1(F, R, folds)
np.save('oof/an__top1.npy', P)
print(f'[B] top1 model alone OOF {score_order(decode(P), R):.4f} top1acc {(P.argmax(1)==t1).mean():.4f}', flush=True)
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + EPS)
best_w, best_sc = 0.0, score_order(decode(Eb), R)
for w in (0.1, 0.2, 0.3, 0.4, 0.5):
    C = (1 - w) * z(Eb) + w * z(P)
    sc = score_order(decode(Eb, first=C.argmax(1)), R)
    print(f'[B] first-site from {1-w:.1f}*gain+{w:.1f}*top1 -> OOF {sc:.4f} top1acc {(C.argmax(1)==t1).mean():.4f}', flush=True)
    if sc > best_sc: best_w, best_sc = w, sc
C = (1 - best_w) * z(Eb) + best_w * z(P) if best_w else z(Eb)
first = C.argmax(1)
print(f'[B] chosen w={best_w} OOF {best_sc:.4f}', flush=True)
order = decode(Eb, first=first)
for (i, j) in [(0, 1), (3, 4)]:
    pp = oof_pair(F, R, folds, i, j)
    sel = np.array([set(o[:2]) == {i, j} for o in order])
    newfirst = first.copy()
    newfirst[sel] = np.where(pp[sel] > 0.5, i, j)
    sc = score_order(decode(Eb, first=newfirst), R)
    acc_before = (first[sel] == t1[sel]).mean() if sel.any() else float('nan')
    acc_after = (newfirst[sel] == t1[sel]).mean() if sel.any() else float('nan')
    print(f'[C] pair {h.SITES[i]}/{h.SITES[j]} rows {sel.sum()} top1acc {acc_before:.3f} -> {acc_after:.3f} OOF {sc:.4f}', flush=True)
    if sc > best_sc: best_sc, first = sc, newfirst
print(f'[FINAL] OOF {best_sc:.4f} top1acc {(first==t1).mean():.4f}  (v3 ref 0.7689, LGB-only ref 0.7640)', flush=True)
