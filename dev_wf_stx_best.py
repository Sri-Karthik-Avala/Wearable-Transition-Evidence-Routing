"""stx best member: site-token transformer (SiteRanker) with the stx-tuned config.

Self-contained: numpy + torch only. fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0) -> (n_te, 5) scores,
higher = rank first. Deterministic given seed (CPU, fixed thread count). Trains CFG["n_seeds"] sub-seeds
derived from `seed` and averages their outputs.
NOTE: CFG currently holds the v1 reproduction config; it is replaced by the tuned config once runs finish.
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
DEVICE = torch.device("cpu")
LEADS = (1.0, 3.0, 5.0, 7.0)

CFG = dict(d=64, layers=2, heads=4, ff=2, drop=0.1, wd=1e-2, epochs=80, lr=2e-3, batch=64, temp=0.15,
           noise=0.10, loss="listnet_mse", out="logsoftmax", mse_w=1.0, mixup=0.0, ema=0.0, n_seeds=3,
           use_seq=True, lead_token=False)


def site_stats(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)
    norms = np.stack([np.linalg.norm(Xs[..., k:k + 3], axis=-1) for k in (0, 3, 6)], -1)
    parts = [Xs.std(2), Xs.mean(2), Xs[:, :, -1] - Xs[:, :, 0], np.abs(np.diff(Xs, axis=2)).mean(2),
             Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2), norms.std(2), norms.mean(2),
             np.abs(np.diff(norms, axis=2)).mean(2)]
    return Xs, np.concatenate(parts, -1).astype(np.float32)


def lead_onehot(lead):
    return np.stack([(lead == l).astype(np.float32) for l in LEADS], 1)


class SiteRanker(nn.Module):
    def __init__(self, n_stat, c):
        super().__init__()
        d, p = c["d"], c["drop"]
        self.use_seq, self.lead_token = c["use_seq"], c["lead_token"]
        self.seq = nn.Sequential(nn.Linear(72, d), nn.GELU(), nn.Dropout(p))
        self.stat = nn.Sequential(nn.Linear(n_stat, d), nn.GELU(), nn.Dropout(p))
        self.site = nn.Parameter(torch.zeros(5, d))
        self.lead = nn.Linear(4, d)
        layer = nn.TransformerEncoderLayer(d, c["heads"], c["ff"] * d, dropout=p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, c["layers"], enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, seq, stat, lead):
        h = self.stat(stat) + self.site[None]
        if self.use_seq:
            h = h + self.seq(seq)
        if self.lead_token:
            h = torch.cat([self.lead(lead)[:, None], h], 1)
            return self.head(self.enc(h)[:, 1:]).squeeze(-1)
        h = h + self.lead(lead)[:, None]
        return self.head(self.enc(h)).squeeze(-1)


def _fit_one(Xtr, Rtr, ltr, Xte, lte, seed, c):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Str, Ftr = site_stats(Xtr)
    Ste, Fte = site_stats(Xte)
    # normalisation stats from the training rows only; each test row is transformed independently
    mu = Ftr.reshape(-1, Ftr.shape[-1]).mean(0)
    sd = Ftr.reshape(-1, Ftr.shape[-1]).std(0) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    seq_tr, st_tr, ld_tr, y = t(Str.reshape(len(Str), 5, 72)), t(Ftr), t(lead_onehot(ltr)), t(Rtr)
    seq_te, st_te, ld_te = t(Ste.reshape(len(Ste), 5, 72)), t(Fte), t(lead_onehot(lte))
    model = SiteRanker(Ftr.shape[-1], c).to(DEVICE)
    ema_model = copy.deepcopy(model) if c["ema"] > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    B = c["batch"]
    steps = c["epochs"] * ((len(Str) + B - 1) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps + 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(c["epochs"]):
        model.train()
        perm = torch.randperm(len(Str), generator=g)
        for i in range(0, len(Str), B):
            b = perm[i:i + B]
            sq = seq_tr[b] + c["noise"] * torch.randn_like(seq_tr[b])
            st, ld, yb = st_tr[b], ld_tr[b], y[b]
            tgt = F.softmax(yb / c["temp"], 1)
            if c["mixup"] > 0:
                lam = float(rng.beta(c["mixup"], c["mixup"]))
                lam = max(lam, 1 - lam)
                p2 = torch.randperm(len(b), generator=g)
                sq, st, ld = lam * sq + (1 - lam) * sq[p2], lam * st + (1 - lam) * st[p2], lam * ld + (1 - lam) * ld[p2]
                tgt, yb = lam * tgt + (1 - lam) * tgt[p2], lam * yb + (1 - lam) * yb[p2]
            s = model(sq, st, ld)
            mse = F.mse_loss(torch.sigmoid(s), yb)
            lnet = -(tgt * F.log_softmax(s, 1)).sum(1).mean()
            if c["loss"] == "listnet_mse":
                loss = lnet + c["mse_w"] * mse
            elif c["loss"] == "gain_mse":
                loss = mse
            elif c["loss"] == "gain_mse_ln":
                loss = mse + 0.1 * lnet
            else:
                raise ValueError(c["loss"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if ema_model is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        pe.mul_(c["ema"]).add_(pm.detach(), alpha=1 - c["ema"])
    m = ema_model if ema_model is not None else model
    m.eval()
    with torch.no_grad():
        s = m(seq_te, st_te, ld_te)
        return (torch.log_softmax(s, 1) if c["out"] == "logsoftmax" else torch.sigmoid(s)).numpy()


def fit_predict(Xtr, Rtr, ltr, Xte, lte, seed=0):
    c = CFG
    subs = [seed * 1000 + k for k in range(c["n_seeds"])]
    return np.mean([_fit_one(Xtr, Rtr, ltr, Xte, lte, s, c) for s in subs], 0).astype(np.float32)


if __name__ == "__main__":
    import wf_harness as h
    d = h.load()
    X, R, lead = d["X"], d["R"], d["lead"]
    S = np.zeros((len(X), 5), np.float32)
    t0 = time.time()
    for tr, va in h.folds(d["groups"]):
        S[va] = fit_predict(X[tr], R[tr], lead[tr], X[va], lead[va], seed=0)
        print(f"  fold done {time.time() - t0:.0f}s", flush=True)
    h.report("stx__best", S, R, lead)
    print(f"boundary_se {h.boundary_se(S, R, d['groups']):.4f}  oof_time {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    Pt = fit_predict(X, R, lead, d["Xt"], d["lead_t"], seed=0)
    print(f"full_train_runtime_sec {time.time() - t1:.0f}  test_pred_shape {Pt.shape}", flush=True)
