import json, sys, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.model_selection import GroupKFold

SITES = ("RWrist", "RUpArm", "Waist", "LThigh", "LAnkle")
G = np.array([1.0, 0.70, 0.45, 0.25, 0.10])
D = 1 / np.log2(np.arange(5) + 2.0)
IDEAL = (G * D).sum(); WORST = (G * D[::-1]).sum()

def rel_matrix(preds):
    R = np.zeros((len(preds), 5))
    for i, p in enumerate(preds):
        for r, s in enumerate(p.split(">")):
            R[i, SITES.index(s)] = G[r]
    return R

def score_from_scores(S, R):
    order = np.argsort(-S, axis=1, kind="stable")
    dcg = (np.take_along_axis(R, order, 1) * D).sum(1)
    return (dcg - WORST) / (IDEAL - WORST)

cache = "dev_nn_cache.npz"
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); X, R, groups, lead = z["X"], z["R"], z["groups"], z["lead"]
else:
    tr = pd.read_csv("train.csv").merge(pd.read_csv("validation_groups.csv"), on="id")
    X = np.stack([np.array(json.loads(s), dtype=np.float32) for s in tr.sensor_sequence]).reshape(-1, 8, 45)
    R = rel_matrix(tr.prediction.values); groups = tr.validation_group.values; lead = tr.lead_offset_s.values.astype(np.float32)
    np.savez(cache, X=X, R=R, groups=groups, lead=lead)
folds = list(GroupKFold(n_splits=5).split(X, groups=groups))
dev = torch.device("cuda:0")

def site_feats(X):
    n = len(X); Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)  # n,5,8,9
    raw = Xs.reshape(n, 5, 72)
    st = [Xs.std(2), Xs.mean(2), np.abs(np.diff(Xs, axis=2)).mean(2), Xs[:, :, -1] - Xs[:, :, 0],
          Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2), np.abs(np.diff(Xs[:, :, -3:], axis=2)).mean(2)]
    st = np.concatenate(st, 2)  # n,5,54
    return np.concatenate([raw, st], 2).astype(np.float32)

class SiteTF(nn.Module):
    def __init__(self, din, d=64, L=2, drop=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        self.site = nn.Parameter(torch.randn(5, d) * 0.02)
        self.lead = nn.Embedding(4, d)
        self.glob = nn.Linear(din * 5, d)
        el = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, L)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x, li):
        h = self.inp(x) + self.site + self.lead(li)[:, None] + self.glob(x.flatten(1))[:, None]
        return self.head(self.enc(h)).squeeze(-1)

class MLP(nn.Module):
    def __init__(self, din, d=256, drop=0.3):
        super().__init__()
        self.lead = nn.Embedding(4, 16)
        self.net = nn.Sequential(nn.Linear(din * 5 + 16, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 5))
    def forward(self, x, li):
        return self.net(torch.cat([x.flatten(1), self.lead(li)], 1))

def run(arch, loss, epochs=80, seeds=3, lr=2e-3, wd=1e-2, noise=0.1, T=0.1, bs=64, d=64, drop=0.1):
    Fx = site_feats(X)
    mu = None
    li_all = np.searchsorted([1., 3., 5., 7.], lead)
    oof = np.zeros_like(R)
    for trn, val in folds:
        m_, s_ = Fx[trn].mean((0, 1), keepdims=True), Fx[trn].std((0, 1), keepdims=True) + 1e-6
        Ftr = torch.tensor((Fx[trn] - m_) / s_, device=dev); Fva = torch.tensor((Fx[val] - m_) / s_, device=dev)
        Rtr = torch.tensor(R[trn], device=dev, dtype=torch.float32)
        ltr = torch.tensor(li_all[trn], device=dev); lva = torch.tensor(li_all[val], device=dev)
        pv = np.zeros((len(val), 5))
        for seed in range(seeds):
            torch.manual_seed(seed)
            net = (SiteTF(Fx.shape[2], d=d, drop=drop) if arch == "tf" else MLP(Fx.shape[2], d=d, drop=drop)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
            steps = epochs * ((len(trn) + bs - 1) // bs)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
            for ep in range(epochs):
                net.train(); perm = torch.randperm(len(trn), device=dev)
                for i in range(0, len(trn), bs):
                    b = perm[i:i + bs]
                    xb = Ftr[b] + noise * torch.randn_like(Ftr[b])
                    out = net(xb, ltr[b])
                    if loss == "listnet":
                        l = -(Fn.softmax(Rtr[b] / T, 1) * Fn.log_softmax(out, 1)).sum(1).mean()
                    else:
                        l = Fn.mse_loss(out, Rtr[b])
                    opt.zero_grad(); l.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad():
                o = net(Fva, lva)
                pv += (Fn.softmax(o, 1) if loss == "listnet" else o).cpu().numpy()
        oof[val] = pv / seeds
    sc = score_from_scores(oof, R)
    return oof, sc

if __name__ == "__main__":
    cfgs = {
        "tf_listnet": dict(arch="tf", loss="listnet"),
        "tf_mse": dict(arch="tf", loss="mse"),
        "mlp_listnet": dict(arch="mlp", loss="listnet", d=256, drop=0.3),
        "mlp_mse": dict(arch="mlp", loss="mse", d=256, drop=0.3),
    }
    sel = sys.argv[1:] or list(cfgs)
    for k in sel:
        oof, sc = run(**cfgs[k])
        np.save(f"dev_nn_oof_{k}.npy", oof)
        print(k, round(sc.mean(), 4), {l: round(sc[lead == l].mean(), 4) for l in [1., 3., 5., 7.]}, flush=True)
