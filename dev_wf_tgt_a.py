"""tgt explorer A: reference per-site LGBM regression vs pairwise (wide + antisymmetric long) vs per-position multiclass."""
import sys, time, warnings
import numpy as np
import lightgbm as lgb
import wf_harness as h
from dev_wf_tgt_common import *
warnings.filterwarnings("ignore")

d = h.load()
X, R, lead, groups = d["X"], d["R"].astype(np.float64), d["lead"], d["groups"]
F = h.folds(groups)
n = len(X)
W = wide(X, lead)
SF = site_feats(X)
rc = rank_class(R)
pos_of = np.argsort(np.argsort(-R, 1, kind="stable"), 1)  # (n,5) position of each site (0=top)
which = sys.argv[1] if len(sys.argv) > 1 else "all"
PRM0 = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2)
PRM = dict(PRM0)
if which == "pos": PRM.update(n_estimators=150, learning_rate=0.06)
res = {}


def rep(name, S):
    res[name] = h.report(f"tgt__{name}", S, R.astype(np.float32), lead)


t0 = time.time()
if which in ("all", "ref"):
    S = np.zeros((n, 5))
    for tr, va in F:
        for s in range(5):
            m = lgb.LGBMRegressor(**PRM, random_state=0).fit(W[tr], R[tr, s])
            S[va, s] = m.predict(W[va])
    rep("ref_siteReg", S)
    print("se", h.boundary_se(S, R, groups), time.time() - t0, flush=True)

if which in ("all", "pw"):
    # (1a) 10 wide pairwise classifiers
    P = {k: np.zeros(n) for k in PAIRS}
    for tr, va in F:
        for (i, j) in PAIRS:
            y = (R[tr, i] > R[tr, j]).astype(int)
            m = lgb.LGBMClassifier(**PRM, random_state=0).fit(W[tr], y)
            P[(i, j)][va] = m.predict_proba(W[va])[:, 1]
    np.save("oof/tgt__pw_wide_P.npy", np.stack([P[k] for k in PAIRS], 1))
    S = np.zeros((n, 5))
    for (i, j) in PAIRS:
        S[:, i] += P[(i, j)]; S[:, j] += 1 - P[(i, j)]
    rep("pw_wide_borda", S)
    lp = pairwise_logp(P)
    for T in (1.0, 0.5):
        rep(f"pw_wide_expgain_T{T}", exp_gain_from_logp(lp * T))
    print("pw pair acc", {k: round(float(((P[k] > .5) == (R[:, k[0]] > R[:, k[1]])).mean()), 3) for k in PAIRS}, flush=True)
    print(time.time() - t0, flush=True)

if which in ("all", "pwl"):
    # (1b) one antisymmetric pairwise model on ordered pairs: [f_i, f_j, f_i - f_j, onehot(i), onehot(j), lead, context]
    ctx = SF.reshape(n, -1)
    OP = [(i, j) for i in range(5) for j in range(5) if i != j]

    def build(idx):
        rows = []
        for (i, j) in OP:
            oh = np.zeros((len(idx), 10), np.float32); oh[:, i] = 1; oh[:, 5 + j] = 1
            rows.append(np.concatenate([SF[idx, i], SF[idx, j], SF[idx, i] - SF[idx, j], oh, lead[idx, None]], 1))
        return np.stack(rows, 1)  # (m,20,F)

    for use_ctx in (False,):
        P = {k: np.zeros(n) for k in PAIRS}
        for tr, va in F:
            A = build(tr); B = build(va)
            y = np.stack([(R[tr, i] > R[tr, j]) for (i, j) in OP], 1).astype(int)
            m = lgb.LGBMClassifier(**{**PRM, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31}, random_state=0)
            m.fit(A.reshape(-1, A.shape[2]), y.reshape(-1))
            pr = m.predict_proba(B.reshape(-1, B.shape[2]))[:, 1].reshape(len(va), 20)
            for (i, j) in PAIRS:
                a = pr[:, OP.index((i, j))]; b = pr[:, OP.index((j, i))]
                P[(i, j)][va] = 0.5 * (a + 1 - b)  # symmetrise
        S = np.zeros((n, 5))
        for (i, j) in PAIRS:
            S[:, i] += P[(i, j)]; S[:, j] += 1 - P[(i, j)]
        rep("pw_long_borda", S)
        lp = pairwise_logp(P)
        for T in (1.0, 0.5):
            rep(f"pw_long_expgain_T{T}", exp_gain_from_logp(lp * T))
    print(time.time() - t0, flush=True)

if which in ("all", "pos"):
    # (2) per-site multiclass over its rank position -> expected gain; plus top-1 multiclass; plus per-position multiclass (which site at k)
    M = np.zeros((n, 5, 5))  # M[:, s, k] = P(site s at position k)
    for tr, va in F:
        for s in range(5):
            m = lgb.LGBMClassifier(**PRM, random_state=0).fit(W[tr], pos_of[tr, s])
            M[va, s] = m.predict_proba(W[va])
    rep("pos_siteRankCls_expgain", M @ G)
    # sinkhorn to doubly-stochastic
    Mk = M.copy()
    for _ in range(50):
        Mk /= Mk.sum(2, keepdims=True); Mk /= Mk.sum(1, keepdims=True)
    rep("pos_siteRankCls_sinkhorn_expgain", Mk @ G)
    Q = np.zeros((n, 5, 5))  # Q[:, k, s] = P(site s at position k)
    for tr, va in F:
        for k in range(5):
            y = np.argsort(-R[tr], 1, kind="stable")[:, k]
            m = lgb.LGBMClassifier(**PRM, random_state=0).fit(W[tr], y)
            Q[va, k] = m.predict_proba(W[va])
    Mq = Q.transpose(0, 2, 1)
    rep("pos_perPosCls_expgain", Mq @ G)
    for _ in range(50):
        Mq = Mq / Mq.sum(2, keepdims=True); Mq = Mq / Mq.sum(1, keepdims=True)
    rep("pos_perPosCls_sinkhorn_expgain", Mq @ G)
    rep("pos_top1_only", Q[:, 0])
    print(time.time() - t0, flush=True)

print(res)
