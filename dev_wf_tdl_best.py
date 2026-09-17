"""KEY=tdl best member: temporal deep model over the 8x45 sequence (self-contained; numpy/torch only).
fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0) -> (n_te,5) expected-gain-like scores (higher = rank first).
CPU, 2 threads, deterministic given seed. __main__ runs grouped 5-fold OOF via wf_harness as tdl__best.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
LEADS = (1.0, 3.0, 5.0, 7.0)
# filled in from the winning dev_wf_tdl_x.py config
CFG = dict(arch="site", enc="conv", mix="attn", stats=True, stat_mode="head", d=64, drop=0.1, layers=2,
           epochs=60, bs=64, lr=2e-3, wd=1e-2, listnet=0.1, T=0.15, Tout=0.3,
           noise=0.1, scale=0.05, sdrop=0.02, shift=0.0)
SEEDS = (0, 1, 2)


def site_stats(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)
    norms = np.stack([np.linalg.norm(Xs[..., k:k + 3], axis=-1) for k in (0, 3, 6)], -1)
    parts = [Xs.std(2), Xs.mean(2), Xs[:, :, -1] - Xs[:, :, 0], np.abs(np.diff(Xs, axis=2)).mean(2),
             Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2), norms.std(2), norms.mean(2),
             np.abs(np.diff(norms, axis=2)).mean(2)]
    return np.concatenate(parts, -1).astype(np.float32)


class CNN(nn.Module):
    def __init__(self, cin=45, d=128, drop=0.2, **_):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(cin, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU(),
                                  nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU(),
                                  nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU())
        self.lead = nn.Embedding(4, 16)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(3 * d + 16, d), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(d, 5))

    def forward(self, x, li, st=None):
        z = self.conv(x.transpose(1, 2))
        return self.head(torch.cat([z.mean(2), z.amax(2), z[:, :, -1], self.lead(li)], 1))


class SiteNet(nn.Module):
    def __init__(self, d=64, drop=0.1, enc="conv", mix="attn", n_stat=0, stat_mode="head", layers=2, **_):
        super().__init__()
        self.enc_t, self.mix_t, self.stat_mode, self.n_stat = enc, mix, stat_mode, n_stat
        if enc == "conv":
            self.enc = nn.Sequential(nn.Conv1d(18, d, 3, padding=1), nn.GELU(), nn.Conv1d(d, d, 3, padding=1), nn.GELU())
            self.pool = nn.Linear(3 * d, d)
        else:
            self.enc = nn.GRU(18, d, batch_first=True)
            self.pool = nn.Linear(2 * d, d)
        self.site = nn.Parameter(torch.randn(5, d) * 0.02)
        self.lead = nn.Embedding(4, d)
        if mix == "attn":
            self.mix = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True), layers)
        else:
            self.ds = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        hd = d
        if n_stat:
            self.stat = nn.Sequential(nn.Linear(n_stat, d), nn.GELU(), nn.Dropout(drop))
            hd = 2 * d if stat_mode == "head" else d
        self.head = nn.Sequential(nn.LayerNorm(hd), nn.Linear(hd, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x, li, st=None):
        B = x.shape[0]
        xs = x.reshape(B, 8, 5, 9).permute(0, 2, 1, 3).reshape(B * 5, 8, 9)
        dx = torch.cat([torch.zeros_like(xs[:, :1]), xs[:, 1:] - xs[:, :-1]], 1)
        z = torch.cat([xs, dx], 2)
        if self.enc_t == "conv":
            e = self.enc(z.transpose(1, 2))
            hh = self.pool(torch.cat([e.mean(2), e.amax(2), e[:, :, -1]], 1))
        else:
            o, hn = self.enc(z)
            hh = self.pool(torch.cat([o.mean(1), hn[-1]], 1))
        hh = hh.reshape(B, 5, -1) + self.site[None] + self.lead(li)[:, None]
        if self.n_stat and self.stat_mode == "in":
            hh = hh + self.stat(st)
        if self.mix_t == "attn":
            hh = self.mix(hh)
        else:
            hh = hh + self.ds(torch.cat([hh, hh.mean(1, keepdim=True).expand_as(hh)], 2))
        if self.n_stat and self.stat_mode == "head":
            hh = torch.cat([hh, self.stat(st)], 2)
        return self.head(hh).squeeze(-1)


def augment(x, g, cfg):
    B = x.shape[0]
    if cfg.get("shift", 0) > 0:
        m = torch.rand(B, generator=g) < cfg["shift"]
        dirn = torch.rand(B, generator=g) < 0.5
        sh = torch.where(dirn[:, None, None], torch.cat([x[:, 1:], x[:, -1:]], 1), torch.cat([x[:, :1], x[:, :-1]], 1))
        x = torch.where(m[:, None, None], sh, x)
    if cfg.get("scale", 0) > 0:
        x = x * torch.exp(cfg["scale"] * torch.randn(B, 1, 45, generator=g))
    if cfg.get("sdrop", 0) > 0:
        m = 1 - (torch.rand(B, 15, generator=g) < cfg["sdrop"]).float().repeat_interleave(3, 1)
        x = x * m[:, None, :]
    if cfg.get("noise", 0) > 0:
        x = x + cfg["noise"] * torch.randn(x.shape, generator=g)
    return x


def _fit_one(Xtr, Rtr, ltr, Xte, lte, seed, cfg):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6  # train-fold stats only
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    st_tr = st_te = None
    if cfg.get("stats"):
        Ftr, Fte = site_stats(A), site_stats(B)
        m2, s2 = Ftr.mean((0, 1), keepdims=True), Ftr.std((0, 1), keepdims=True) + 1e-6
        st_tr, st_te = torch.tensor((Ftr - m2) / s2), torch.tensor((Fte - m2) / s2)
    if cfg.get("inp") == "center":  # per-row self-normalisation
        A, B = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    xtr, xte = torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)
    litr = torch.tensor(np.searchsorted(LEADS, ltr)); lite = torch.tensor(np.searchsorted(LEADS, lte))
    y = torch.tensor(Rtr, dtype=torch.float32)
    if cfg["arch"] == "cnn":
        net = CNN(d=cfg.get("d", 128), drop=cfg.get("drop", 0.2))
    else:
        net = SiteNet(d=cfg["d"], drop=cfg["drop"], enc=cfg["enc"], mix=cfg["mix"],
                      n_stat=54 if cfg.get("stats") else 0, stat_mode=cfg.get("stat_mode", "head"), layers=cfg["layers"])
    ep, bs, lr = cfg["epochs"], cfg["bs"], cfg["lr"]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg["wd"])
    nb = (len(xtr) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=ep * nb + 1, pct_start=0.25)
    tgt = F.softmax(y / cfg["T"], 1)
    for _ in range(ep):
        net.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            out = net(augment(xtr[b], g, cfg), litr[b], None if st_tr is None else st_tr[b])
            loss = F.mse_loss(out, y[b])
            if cfg["listnet"] > 0:
                loss = loss + cfg["listnet"] * -(tgt[b] * F.log_softmax(out / cfg["Tout"], 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        return net(xte, lite, st_te).numpy()


def fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0):
    return np.mean([_fit_one(Xtr, Rtr, ltr, Xte, lte, seed * 100 + s, CFG) for s in SEEDS], 0).astype(np.float32)


def _retry(fn, tries=40, wait=15):
    """dev-only: host commit memory is exhausted by other jobs; rerun a fit from scratch on allocator OOM."""
    for k in range(tries):
        try:
            return fn()
        except (RuntimeError, MemoryError) as e:
            if "memory" not in str(e).lower() or k == tries - 1:
                raise
            print(f"  OOM, retry {k + 1} in {wait}s", flush=True)
            time.sleep(wait)


if __name__ == "__main__":
    import wf_harness as h
    d = h.load()
    S = np.zeros_like(d["R"])
    t0 = time.time()
    for tr, va in h.folds(d["groups"]):
        S[va] = _retry(lambda: fit_predict(d["X"][tr], d["R"][tr], d["lead"][tr], d["X"][va], d["lead"][va], seed=0))
    print(f"cv time {time.time() - t0:.0f}s", flush=True)
    h.report("tdl__best", S, d["R"], d["lead"])
    print("boundary_se", round(h.boundary_se(S, d["R"], d["groups"]), 4))
    t0 = time.time()
    fit_predict(d["X"], d["R"], d["lead"], d["Xt"], d["lead_t"], seed=0)
    print(f"full-train runtime {time.time() - t0:.0f}s", flush=True)
