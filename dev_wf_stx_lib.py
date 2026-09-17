"""stx explorer: configurable site-token transformer (v1 SiteRanker reproduction + tuning knobs)."""
import sys, time, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
DEVICE = torch.device("cpu")
LEADS = (1.0, 3.0, 5.0, 7.0)

BASE = dict(d=64, layers=2, heads=4, ff=2, drop=0.1, wd=1e-2, epochs=80, lr=2e-3, batch=64, temp=0.15,
            noise=0.10, loss="listnet_mse", out="logsoftmax", mse_w=1.0, site_drop=0.0, mixup=0.0,
            ema=0.0, seeds=(0, 1, 2), use_seq=True, lead_token=False, stat_noise=0.0, tok_drop=0.0)


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

    def forward(self, seq, stat, lead, tok_mask=None):
        h = self.stat(stat) + self.site[None]
        if self.use_seq:
            h = h + self.seq(seq)
        if tok_mask is not None:
            h = h * tok_mask[..., None]
        if self.lead_token:
            h = torch.cat([self.lead(lead)[:, None], h], 1)
            return self.head(self.enc(h)[:, 1:]).squeeze(-1)
        h = h + self.lead(lead)[:, None]
        return self.head(self.enc(h)).squeeze(-1)


def fit_predict_one(Xtr, Rtr, ltr, Xte, lte, seed, c):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Str, Ftr = site_stats(Xtr)
    Ste, Fte = site_stats(Xte)
    mu = Ftr.reshape(-1, Ftr.shape[-1]).mean(0)
    sd = Ftr.reshape(-1, Ftr.shape[-1]).std(0) + 1e-6
    Ftr = (Ftr - mu) / sd
    Fte = (Fte - mu) / sd
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
            st = st_tr[b]
            if c["stat_noise"] > 0:
                st = st + c["stat_noise"] * torch.randn_like(st)
            ld, yb = ld_tr[b], y[b]
            tgt = F.softmax(yb / c["temp"], 1)
            if c["mixup"] > 0:
                lam = float(np.random.beta(c["mixup"], c["mixup"]))
                lam = max(lam, 1 - lam)
                p2 = torch.randperm(len(b), generator=g)
                sq = lam * sq + (1 - lam) * sq[p2]
                st = lam * st + (1 - lam) * st[p2]
                ld = lam * ld + (1 - lam) * ld[p2]
                tgt = lam * tgt + (1 - lam) * tgt[p2]
                yb = lam * yb + (1 - lam) * yb[p2]
            tm = None
            if c["tok_drop"] > 0:
                tm = (torch.rand(len(b), 5, generator=g) > c["tok_drop"]).float()
            s = model(sq, st, ld, tm)
            if c["site_drop"] > 0:
                # drop sites from the list: exclude from softmax, zero their mse weight
                keep = (torch.rand(len(b), 5, generator=g) > c["site_drop"])
                keep[keep.sum(1) < 2] = True
                neg = torch.where(keep, torch.zeros_like(s), torch.full_like(s, -1e4))
                tl = F.softmax(yb / c["temp"] + neg, 1)
                ls = F.log_softmax(s + neg, 1)
                kf = keep.float()
                lnet = -(tl * ls * kf).sum(1).mean()
                mse = (((torch.sigmoid(s) - yb) ** 2) * kf).sum() / kf.sum()
            else:
                lnet = -(tgt * F.log_softmax(s, 1)).sum(1).mean()
                mse = F.mse_loss(torch.sigmoid(s), yb)
            if c["loss"] == "listnet_mse":
                loss = lnet + c["mse_w"] * mse
            elif c["loss"] == "gain_mse":
                loss = mse + c["mse_w"] * 0.0 * lnet
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
        if c["out"] == "logsoftmax":
            return torch.log_softmax(s, 1).numpy()
        return torch.sigmoid(s).numpy()


def fit_predict(Xtr, Rtr, ltr, Xte, lte, c, seeds=None):
    seeds = c["seeds"] if seeds is None else seeds
    return np.mean([fit_predict_one(Xtr, Rtr, ltr, Xte, lte, s, c) for s in seeds], 0)


def _retry(fn, tries=20):
    # dev-only: survive transient host OOM from other jobs; fits are seeded so a retry is identical
    for k in range(tries):
        try:
            return fn()
        except RuntimeError as e:
            if "not enough memory" not in str(e) or k == tries - 1:
                raise
            print("  OOM, retrying in 15s", flush=True)
            time.sleep(15)


def run(name, overrides):
    import wf_harness as h
    c = dict(BASE)
    c.update(overrides)
    d = h.load()
    X, R, lead = d["X"], d["R"], d["lead"]
    S = np.zeros((len(X), 5), np.float32)
    S1 = np.zeros((len(X), 5), np.float32)  # seed-0 only
    t0 = time.time()
    for tr, va in h.folds(d["groups"]):
        outs = [_retry(lambda: fit_predict_one(X[tr], R[tr], lead[tr], X[va], lead[va], s, c)) for s in c["seeds"]]
        S[va] = np.mean(outs, 0)
        S1[va] = outs[0]
        print(f"  {name} fold done {time.time() - t0:.0f}s fold_oof={h.row_scores(S[va], R[va]).mean():.4f}", flush=True)
    dt = time.time() - t0
    sc = h.report(f"stx__{name}", S, R, lead)
    sc1 = h.report(f"stx__{name}_s0", S1, R, lead, save=False)
    print(f"RESULT {name} oof={sc:.4f} seed0={sc1:.4f} se={h.boundary_se(S, R, d['groups']):.4f} "
          f"time={dt:.0f}s cfg={json.dumps(overrides)}", flush=True)
    return sc


PRESETS = {
    "v1s0": {"seeds": (0,)},
    "v1gains0": {"loss": "gain_mse", "out": "sigmoid", "seeds": (0,)},
}

if __name__ == "__main__":
    # usage: python dev_wf_stx_lib.py name '{"json":"overrides"}' [name2 '{...}' ...]
    #    or: python dev_wf_stx_lib.py @preset [@preset ...]   (quote-free, for WMI launches)
    args = sys.argv[1:]
    if args and all(a.startswith("@") for a in args):
        for a in args:
            run(a[1:], dict(PRESETS[a[1:]]))
        sys.exit(0)
    for i in range(0, len(args), 2):
        ov = json.loads(args[i + 1])
        if "seeds" in ov:
            ov["seeds"] = tuple(ov["seeds"])
        run(args[i], ov)
