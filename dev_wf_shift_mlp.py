"""(3b) same shift-robust transforms inside a small site-shared MLP (ListNet + MSE), CPU 2 threads."""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import wf_harness as h
import dev_wf_shift_lib as L

torch.set_num_threads(2)
d = h.load()
X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
FOLDS = h.folds(groups)
LEADS = (1.0, 3.0, 5.0, 7.0)

SETS = {
    "v1like": {"dyn", "clast", "lev", "maglev", "rnorm", "magnm"},
    "nomaglev": {"dyn", "clast", "lev", "rnorm"},
    "center": {"dyn", "clast", "cnorm"},
    "center_rnorm": {"dyn", "clast", "cnorm", "rnorm"},
    "center_inv": {"dyn", "clast", "cnorm", "rnorm", "ang"},
}


def site_input(X, lead, bl):
    Fs = L.site_feats(X, bl)
    rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
    ctx = np.repeat(Fs.mean(1, keepdims=True), 5, 1)
    lo = np.stack([(lead == l).astype(np.float32) for l in LEADS], 1)
    return np.concatenate([Fs, rel, ctx], -1), lo


class Net(nn.Module):
    def __init__(self, k, hdim=128):
        super().__init__()
        self.inp = nn.Linear(k, hdim)
        self.site = nn.Parameter(torch.zeros(5, hdim))
        self.lead = nn.Linear(4, hdim)
        self.body = nn.Sequential(nn.GELU(), nn.Dropout(0.1), nn.Linear(hdim, hdim), nn.GELU(), nn.Dropout(0.1))
        self.pool = nn.Linear(hdim, hdim)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hdim, 1))

    def forward(self, x, lo):
        z = self.body(self.inp(x) + self.site[None] + self.lead(lo)[:, None])
        z = z + self.pool(z.mean(1, keepdim=True))  # cross-site context
        return self.head(z).squeeze(-1)


def fit_predict(Xtr, Rtr, ltr, Xte, lte, bl, seed=0, epochs=60, aug=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if aug is not None:
        Xtr, Rtr, ltr = L.augment(Xtr, Rtr, ltr, rng, **aug)
    A, la = site_input(Xtr, ltr, bl)
    B, lb = site_input(Xte, lte, bl)
    mu = A.reshape(-1, A.shape[-1]).mean(0); sd = A.reshape(-1, A.shape[-1]).std(0) + 1e-6
    A = np.clip((A - mu) / sd, -6, 6); B = np.clip((B - mu) / sd, -6, 6)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    A, la, B, lb, y = t(A), t(la), t(B), t(lb), t(Rtr)
    tgt = Fn.softmax(y / 0.15, 1)
    net = Net(A.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    bs = 64; nb = (len(A) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * nb + 1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(A), generator=g)
        for i in range(nb):
            b = perm[i * bs:(i + 1) * bs]
            s = net(A[b] + 0.05 * torch.randn(A[b].shape, generator=g), la[b])
            loss = -(tgt[b] * Fn.log_softmax(s, 1)).sum(1).mean() + Fn.mse_loss(torch.sigmoid(s), y[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        return torch.log_softmax(net(B, lb), 1).numpy()


def oof(bl, seeds=(0, 1), aug=None):
    S = np.zeros_like(R)
    for tr, va in FOLDS:
        S[va] = np.mean([fit_predict(X[tr], R[tr], lead[tr], X[va], lead[va], bl, s, aug=aug) for s in seeds], 0)
    return S


if __name__ == "__main__":
    t0 = time.time()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SETS)
    aug = None
    if len(sys.argv) > 2 and sys.argv[2] == "scale":
        aug = dict(copies=1, scale=(0.85, 1.4))
    for nm in names:
        S = oof(SETS[nm], aug=aug)
        h.report(f"shift__mlp_{nm}" + ("_scale" if aug else ""), S, R, lead)
        print(f"  t={time.time() - t0:.0f}s", flush=True)
