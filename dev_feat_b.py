import sys
sys.argv = ["x", "none"]
exec(open("dev_feat_a.py").read())
fr = rich_site_feats(Xs)
d = np.abs(np.diff(Xs, axis=1))
fr["inv_std"] = 1.0 / (Xs.std(1) + 0.05)
fr["inv_lastdiff"] = 1.0 / (d[:, -1] + 0.05)
fr["inv_last3std"] = 1.0 / (Xs[:, -3:].std(1) + 0.05)
fr["qmin"] = d.min(1)
fr["lastdiff"] = d[:, -1]
F = wide(fr, extra=True)
inv = np.stack([fr[k].mean(-1) for k in ["inv_std", "inv_lastdiff", "inv_last3std", "lastdiff"]], -1)
inv_rel = inv / (inv.mean(1, keepdims=True) + 1e-6)
F2 = np.concatenate([F, inv.reshape(n, -1), inv_rel.reshape(n, -1)], 1)
run_wide(F2, "wide_rich+summ+inv")
