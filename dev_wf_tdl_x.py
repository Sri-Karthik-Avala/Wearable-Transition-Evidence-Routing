"""KEY=tdl: temporal deep models over the 8x45 sequence. CPU only, 2 threads.
usage: python -u dev_wf_tdl_x.py cfg1 cfg2 ...
"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wf_harness as h

torch.set_num_threads(2)
LEADS = (1.0, 3.0, 5.0, 7.0)


def site_stats(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)
    norms = np.stack([np.linalg.norm(Xs[..., k:k + 3], axis=-1) for k in (0, 3, 6)], -1)
    parts = [Xs.std(2), Xs.mean(2), Xs[:, :, -1] - Xs[:, :, 0], np.abs(np.diff(Xs, axis=2)).mean(2),
             Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2), norms.std(2), norms.mean(2),
             np.abs(np.diff(norms, axis=2)).mean(2)]
    return np.concatenate(parts, -1).astype(np.float32)  # n,5,54


# ---------------------------------------------------------------- models
class CNN(nn.Module):
    """1D CNN over time on all 45 channels, 5-output head."""
    def __init__(self, cin=45, d=128, drop=0.2, **_):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(cin, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU(),
                                  nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU(),
                                  nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU())
        self.lead = nn.Embedding(4, 16)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(3 * d + 16, d), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(d, 5))

    def forward(self, x, li, st=None):  # x B,8,45
        z = self.conv(x.transpose(1, 2))
        f = torch.cat([z.mean(2), z.amax(2), z[:, :, -1], self.lead(li)], 1)
        return self.head(f)


class SiteNet(nn.Module):
    """shared per-site temporal encoder + site/lead embedding + cross-site mixing + per-site head."""
    def __init__(self, cin=9, d=64, drop=0.1, enc="conv", mix="attn", n_stat=0, stat_mode="head", layers=2, **_):
        super().__init__()
        self.enc_t, self.mix_t, self.stat_mode = enc, mix, stat_mode
        c = 2 * cin  # raw + first difference
        if enc == "conv":
            self.enc = nn.Sequential(nn.Conv1d(c, d, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(d, d, 3, padding=1), nn.GELU())
            self.pool = nn.Linear(3 * d, d)
        else:
            self.enc = nn.GRU(c, d, batch_first=True)
            self.pool = nn.Linear(2 * d, d)
        self.site = nn.Parameter(torch.randn(5, d) * 0.02)
        self.lead = nn.Embedding(4, d)
        if mix == "attn":
            el = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
            self.mix = nn.TransformerEncoder(el, layers)
        else:
            self.ds = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        self.n_stat = n_stat
        hd = d
        if n_stat:
            self.stat = nn.Sequential(nn.Linear(n_stat, d), nn.GELU(), nn.Dropout(drop))
            if stat_mode == "head":
                hd = 2 * d
        self.head = nn.Sequential(nn.LayerNorm(hd), nn.Linear(hd, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x, li, st=None):
        B = x.shape[0]
        xs = x.reshape(B, 8, 5, 9).permute(0, 2, 1, 3).reshape(B * 5, 8, 9)
        dx = torch.cat([torch.zeros_like(xs[:, :1]), xs[:, 1:] - xs[:, :-1]], 1)
        z = torch.cat([xs, dx], 2)  # B5,8,18
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


# ---------------------------------------------------------------- training
def augment(x, g, cfg):
    B = x.shape[0]
    if cfg.get("shift", 0) > 0:
        m = torch.rand(B, generator=g) < cfg["shift"]
        if m.any():
            dirn = torch.rand(B, generator=g) < 0.5
            left = torch.cat([x[:, 1:], x[:, -1:]], 1)   # drop first, repeat last
            right = torch.cat([x[:, :1], x[:, :-1]], 1)  # repeat first, drop last
            sh = torch.where(dirn[:, None, None], left, right)
            x = torch.where(m[:, None, None], sh, x)
    if cfg.get("scale", 0) > 0:
        x = x * torch.exp(cfg["scale"] * torch.randn(B, 1, 45, generator=g))
    if cfg.get("sdrop", 0) > 0:  # zero one sensor triplet of one site
        m = (torch.rand(B, 15, generator=g) < cfg["sdrop"]).float()
        m = 1 - m.repeat_interleave(3, 1)
        x = x * m[:, None, :]
    if cfg.get("noise", 0) > 0:
        x = x + cfg["noise"] * torch.randn(x.shape, generator=g)
    return x


def prep(X, mode):
    if mode == "center":  # per-row self-normalisation: remove each row's own per-channel mean
        return X - X.mean(1, keepdims=True)
    if mode == "both":
        return np.concatenate([X, X - X.mean(1, keepdims=True)], 2)
    return X


def fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0, cfg=None):
    cfg = dict(cfg or {})
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    st_tr = st_te = None
    if cfg.get("stats"):
        Ftr, Fte = site_stats(A), site_stats(B)
        m2, s2 = Ftr.mean((0, 1), keepdims=True), Ftr.std((0, 1), keepdims=True) + 1e-6
        st_tr, st_te = torch.tensor((Ftr - m2) / s2), torch.tensor((Fte - m2) / s2)
    A, B = prep(A, cfg.get("inp", "raw")), prep(B, cfg.get("inp", "raw"))
    xtr, xte = torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)
    litr = torch.tensor(np.searchsorted(LEADS, ltr)); lite = torch.tensor(np.searchsorted(LEADS, lte))
    y = torch.tensor(Rtr, dtype=torch.float32)
    arch = cfg.get("arch", "site")
    if arch == "cnn":
        net = CNN(cin=A.shape[2], d=cfg.get("d", 128), drop=cfg.get("drop", 0.2))
    else:
        cin = 9 if A.shape[2] == 45 else 18
        net = SiteNet(cin=cin, d=cfg.get("d", 64), drop=cfg.get("drop", 0.1), enc=cfg.get("enc", "conv"),
                      mix=cfg.get("mix", "attn"), n_stat=(54 if cfg.get("stats") else 0),
                      stat_mode=cfg.get("stat_mode", "head"), layers=cfg.get("layers", 2))
        if cin == 18:  # 'both' input: site view of 90 channels -> (5 sites x 18)
            xtr = torch.cat([xtr[:, :, :45].reshape(-1, 8, 5, 9), xtr[:, :, 45:].reshape(-1, 8, 5, 9)], 3).reshape(-1, 8, 90)
            xte = torch.cat([xte[:, :, :45].reshape(-1, 8, 5, 9), xte[:, :, 45:].reshape(-1, 8, 5, 9)], 3).reshape(-1, 8, 90)
    if arch != "cnn" and xtr.shape[2] == 90:
        net._site_ch = 18
    ep, bs, lr = cfg.get("epochs", 60), cfg.get("bs", 64), cfg.get("lr", 2e-3)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg.get("wd", 1e-2))
    nb = (len(xtr) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=ep * nb + 1, pct_start=0.25)
    lw, T = cfg.get("listnet", 0.1), cfg.get("T", 0.15)
    tgt = F.softmax(y / T, 1)
    for _ in range(ep):
        net.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = augment(xtr[b], g, cfg) if arch == "cnn" or xtr.shape[2] == 45 else augment90(xtr[b], g, cfg)
            out = net(xb, litr[b], None if st_tr is None else st_tr[b])
            loss = F.mse_loss(out, y[b])
            if lw > 0:
                loss = loss + lw * -(tgt[b] * F.log_softmax(out / cfg.get("Tout", 0.3), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        return net(xte, lite, st_te).numpy()


def augment90(x, g, cfg):
    B = x.shape[0]
    c2 = {k: v for k, v in cfg.items() if k in ("shift",)}
    x = augment(x.reshape(B, 8, 90), g, {**c2}) if False else x
    if cfg.get("noise", 0) > 0:
        x = x + cfg["noise"] * torch.randn(x.shape, generator=g)
    return x


CFGS = {
    # (1) 1D CNN on all 45 channels
    "cnn": dict(arch="cnn", noise=0.1, scale=0.05, sdrop=0.02, shift=0.0),
    # (2) per-site shared encoder + cross-site attention
    "site_conv_attn": dict(arch="site", enc="conv", mix="attn", noise=0.1, scale=0.05, sdrop=0.02),
    "site_gru_attn": dict(arch="site", enc="gru", mix="attn", noise=0.1, scale=0.05, sdrop=0.02),
    "site_conv_ds": dict(arch="site", enc="conv", mix="ds", noise=0.1, scale=0.05, sdrop=0.02),
    # (3) + handcrafted per-site stats
    "site_conv_attn_st": dict(arch="site", enc="conv", mix="attn", stats=True, noise=0.1, scale=0.05, sdrop=0.02),
    "site_conv_attn_stin": dict(arch="site", enc="conv", mix="attn", stats=True, stat_mode="in", noise=0.1, scale=0.05, sdrop=0.02),
}


def retry(fn, tries=40, wait=15):
    """system commit is exhausted by other jobs: retry a seed-fit on allocator OOM instead of dying."""
    for k in range(tries):
        try:
            return fn()
        except (RuntimeError, MemoryError) as e:
            if "memory" not in str(e).lower() or k == tries - 1:
                raise
            print(f"  OOM, retry {k + 1} in {wait}s", flush=True)
            time.sleep(wait)


def run_cfg(name, cfg, seeds=(0, 1, 2)):
    d = h.load()
    X, R, lead = d["X"], d["R"], d["lead"]
    S = np.zeros_like(R)
    t0 = time.time()
    for k, (tr, va) in enumerate(h.folds(d["groups"])):
        S[va] = np.mean([retry(lambda: fit_predict(X[tr], R[tr], lead[tr], X[va], lead[va], seed=s, cfg=cfg))
                         for s in seeds], 0)
        print(f"  {name} fold {k} {time.time() - t0:.0f}s fold-score {h.row_scores(S[va], R[va]).mean():.4f}", flush=True)
    sc = h.report(f"tdl__{name}", S, R, lead)
    print(f"  {name} boundary_se {h.boundary_se(S, R, d['groups']):.4f} time {time.time() - t0:.0f}s", flush=True)
    return sc


if __name__ == "__main__":
    import json
    for arg in sys.argv[1:]:
        if "=" in arg:  # name={json overrides on base}
            name, js = arg.split("=", 1)
            base, over = js.split("+", 1) if "+" in js else (js, "{}")
            cfg = {**CFGS[base], **json.loads(over)}
        else:
            name, cfg = arg, CFGS[arg]
        seeds = tuple(range(cfg.pop("nseeds", 3)))
        run_cfg(name, cfg, seeds)
