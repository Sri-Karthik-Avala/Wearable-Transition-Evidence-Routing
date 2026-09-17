"""Shared per-site tabular features for the tab explorer (per-row only; no cross-row stats)."""
import numpy as np

LEADS = (1.0, 3.0, 5.0, 7.0)


def _rank(v):
    return np.argsort(np.argsort(v, 1, kind="stable"), 1, kind="stable").astype(np.float32)


def site_feats(X):
    """X (n,8,45) -> dict of (n,5,9) per-site-per-channel features."""
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).astype(np.float32)
    d = np.diff(Xs, axis=1)
    ad = np.abs(d)
    mu = Xs.mean(1)
    f = {
        "std": Xs.std(1),
        "mean": mu,
        "delta": Xs[:, -1] - Xs[:, 0],
        "absdiff": ad.mean(1),
        "last": Xs[:, -1],
        "late": Xs[:, -2:].mean(1) - Xs[:, :-2].mean(1),
        "lastdev": np.abs(Xs[:, -1] - mu),
        "min": Xs.min(1),
        "max": Xs.max(1),
        "lastdiff": d[:, -1],
        "std_last3": Xs[:, -3:].std(1),
    }
    return f


SUMM_KEYS = ("std", "absdiff", "lastdev", "std_last3")


def site_summary(f):
    """per-site channel-mean summaries: raw, / 5-site mean, rank among 5 sites. Also per sensor (acc/gyro/mag)."""
    out = []
    for k in SUMM_KEYS:
        v = np.abs(f[k]) if k == "lastdiff" else f[k]
        chans = [v.mean(-1)] + [v[..., 3 * j:3 * j + 3].mean(-1) for j in range(3)]
        for c in chans:
            out.append(c)
            out.append(c / (c.mean(1, keepdims=True) + 1e-6))
            out.append(_rank(c))
    return np.stack(out, -1).astype(np.float32)  # n,5,S


def lead_oh(lead):
    return np.stack([(lead == l).astype(np.float32) for l in LEADS], 1)


def wide(X, lead, keys=None):
    f = site_feats(X)
    keys = keys or list(f.keys())
    n = len(X)
    parts = [f[k].reshape(n, -1) for k in keys]
    parts.append(site_summary(f).reshape(n, -1))
    parts.append(lead[:, None].astype(np.float32))
    return np.concatenate(parts, 1).astype(np.float32)


def summary_only(X, lead):
    f = site_feats(X)
    n = len(X)
    return np.concatenate([site_summary(f).reshape(n, -1), lead[:, None]], 1).astype(np.float32)


def long(X, lead, keys=None):
    """(n,5,F): site id, lead, own per-channel feats, own summary, all-site summaries, all-site channel means."""
    f = site_feats(X)
    keys = keys or list(f.keys())
    n = len(X)
    per = np.concatenate([f[k] for k in keys], -1)  # n,5,K
    summ = site_summary(f)  # n,5,S
    ctx = np.concatenate([summ.reshape(n, -1), f["mean"].reshape(n, -1)], 1)
    rows = []
    for s in range(5):
        sid = np.full((n, 1), s, np.float32)
        rows.append(np.concatenate([sid, lead[:, None], per[:, s], summ[:, s], ctx], 1))
    return np.stack(rows, 1).astype(np.float32)


def rowz(S):
    S = np.asarray(S, dtype=np.float64)
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
