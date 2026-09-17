import itertools
import numpy as np
import wf_harness as h

G = h.GAINS.astype(np.float64)
PERMS = np.array(list(itertools.permutations(range(5))))  # (120,5): perm[k] = site at position k
# gain matrix per permutation in site order: PG[p, s] = gain of site s under perm p
PG = np.zeros((120, 5))
for p, perm in enumerate(PERMS):
    for k, s in enumerate(perm):
        PG[p, s] = G[k]
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def site_feats(X):
    """Per-site, per-channel stats over the 8 frames -> (n,5,9*6)."""
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9)
    f = [Xs.std(1), Xs.mean(1), Xs[:, -1] - Xs[:, 0], np.abs(np.diff(Xs, axis=1)).mean(1),
         Xs[:, -2:].mean(1) - Xs[:, :-2].mean(1), Xs[:, -1]]
    return np.concatenate(f, -1).astype(np.float32)


def wide(X, lead):
    sf = site_feats(X)
    return np.concatenate([sf.reshape(len(X), -1), lead[:, None]], 1).astype(np.float32)


def rank_class(R):
    """perm index of each row's true ranking."""
    tgt = np.argsort(-R, 1, kind="stable")
    lut = {tuple(p): i for i, p in enumerate(PERMS)}
    return np.array([lut[tuple(t)] for t in tgt])


def exp_gain_from_logp(logp):
    """logp (n,120) -> expected gains (n,5)."""
    logp = logp - logp.max(1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(1, keepdims=True)
    return p @ PG


def best_perm_scores(p):
    """exact expected-NDCG-optimal permutation from p(r) (n,120): returns site scores implementing that perm."""
    # expected DCG of candidate c = sum_r p(r) * sum_k PG[r, PERMS[c,k]] * DISC[k]
    DISC = h.DISC
    cand = np.zeros((120, 120))  # cand[c, r] = DCG of candidate c under truth r
    for c, perm in enumerate(PERMS):
        cand[c] = (PG[:, perm] * DISC).sum(1)
    e = p @ cand.T  # (n,120)
    best = e.argmax(1)
    S = np.zeros((len(p), 5))
    for k in range(5):
        S[np.arange(len(p)), PERMS[best, k]] = 5 - k
    return S


def pairwise_logp(P):
    """P dict (i,j)->(n,) prob site i above site j -> logp over perms under product-of-pairs model."""
    n = len(next(iter(P.values())))
    lp = np.zeros((n, 120))
    for c, perm in enumerate(PERMS):
        pos = np.argsort(perm)
        for (i, j) in PAIRS:
            pr = np.clip(P[(i, j)], 1e-4, 1 - 1e-4)
            lp[:, c] += np.log(pr) if pos[i] < pos[j] else np.log(1 - pr)
    return lp
