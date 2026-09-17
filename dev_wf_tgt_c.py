"""tgt explorer C: Plackett-Luce listwise loss vs gain-MSE control in the same small site-shared MLP (CPU)."""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import wf_harness as h
from dev_wf_tgt_common import *
torch.set_num_threads(2)

d = h.load()
X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
F = h.folds(groups)
n = len(X)
SF = site_feats(X)  # (n,5,54)
order = np.argsort(-R, 1, kind="stable")


class Net(nn.Module):
    def __init__(self, fin, hd=64):
        super().__init__()
        self.emb = nn.Embedding(5, 8)
        self.enc = nn.Sequential(nn.Linear(fin + 8 + 1, hd), nn.GELU(), nn.Dropout(0.1), nn.Linear(hd, hd), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * hd, hd), nn.GELU(), nn.Dropout(0.1), nn.Linear(hd, 1))

    def forward(self, f, l):
        b = f.shape[0]
        e = self.emb.weight[None].expand(b, 5, 8)
        z = self.enc(torch.cat([f, e, l[:, None, None].expand(b, 5, 1)], -1))
        c = z.mean(1, keepdim=True).expand_as(z)
        return self.head(torch.cat([z, c, z - c], -1)).squeeze(-1)


def pl_loss(s, ordr):
    so = torch.gather(s, 1, ordr)  # scores in true order
    # log-sum-exp over the suffix
    lse = torch.logcumsumexp(so.flip(1), 1).flip(1)
    return (lse - so)[:, :4].sum(1).mean()


def fit_predict(Ftr, Rtr, ltr, Fte, lte, loss, seed=0, epochs=80):
    torch.manual_seed(seed); np.random.seed(seed)
    mu = Ftr.reshape(-1, Ftr.shape[2]).mean(0); sd = Ftr.reshape(-1, Ftr.shape[2]).std(0) + 1e-6
    ft = torch.tensor((Ftr - mu) / sd, dtype=torch.float32); fe = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
    lt = torch.tensor(ltr / 7.0, dtype=torch.float32); le = torch.tensor(lte / 7.0, dtype=torch.float32)
    Rt = torch.tensor(Rtr, dtype=torch.float32)
    Ot = torch.tensor(np.argsort(-Rtr, 1, kind="stable"), dtype=torch.long)
    net = Net(Ftr.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(ft), generator=g)
        for k in range(0, len(ft), 64):
            b = perm[k:k + 64]
            s = net(ft[b], lt[b])
            if loss == "pl":
                L = pl_loss(s, Ot[b])
            elif loss == "mse":
                L = ((s - Rt[b]) ** 2).mean()
            else:  # pl+mse (mse on the same score, scaled)
                L = pl_loss(s, Ot[b]) + 5.0 * ((s - s.mean(1, keepdim=True) - (Rt[b] - Rt[b].mean(1, keepdim=True))) ** 2).mean()
            opt.zero_grad(); L.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return net(fe, le).numpy()


t0 = time.time()
losses = sys.argv[1:] or ["mse", "pl", "plmse"]
for loss in losses:
    S = np.zeros((n, 5))
    for tr, va in F:
        S[va] = np.mean([fit_predict(SF[tr], R[tr], lead[tr], SF[va], lead[va], loss, seed=s) for s in range(2)], 0)
    h.report(f"tgt__mlp_{loss}", S, R, lead)
    print(loss, round(time.time() - t0, 1), flush=True)
