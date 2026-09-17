"""tgt explorer B: ranking-class models (kNN, softmax over 120 perms) + Plackett-Luce vs MSE MLP + decode ceilings."""
import sys, time, warnings
import numpy as np
import torch
import wf_harness as h
from dev_wf_tgt_common import *
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

d = h.load()
X, R, lead, groups = d["X"], d["R"].astype(np.float64), d["lead"], d["groups"]
F = h.folds(groups)
n = len(X)
W = wide(X, lead)
rc = rank_class(R)
order = np.argsort(-R, 1, kind="stable")
which = sys.argv[1] if len(sys.argv) > 1 else "all"
res = {}


def rep(name, S):
    res[name] = h.report(f"tgt__{name}", S, R.astype(np.float32), lead)


def std_fit(A, tr):
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
    return (A - mu) / sd


t0 = time.time()
if which in ("all", "ceil"):
    # decode ceilings on the reference OOF
    S = np.load("oof/tgt__ref_siteReg.npy").astype(np.float64)
    prior = np.tile(R.mean(0), (n, 1))
    rep("ceil_prior", prior)
    for k in (1, 2, 3):
        Sk = S.copy()
        big = 100.0
        for q in range(k):
            Sk[np.arange(n), order[:, q]] = big - q  # oracle top-k, model orders the rest
        res[f"ceil_oracleTop{k}_modelRest"] = float(h.row_scores(Sk, R).mean())
    # model top-1 correct? score split
    top1_ok = S.argmax(1) == order[:, 0]
    res["model_top1_acc"] = float(top1_ok.mean())
    res["score_when_top1_ok"] = float(h.row_scores(S, R)[top1_ok].mean())
    res["score_when_top1_wrong"] = float(h.row_scores(S, R)[~top1_ok].mean())
    # ranking-class coverage ceiling: best class seen in train fold (hard-class decode)
    ceil = np.zeros(n)
    for tr, va in F:
        seen = np.unique(rc[tr])
        cand = PG[seen]  # (m,5)
        sc = np.stack([h.row_scores(np.tile(c, (len(va), 1)), R[va]) for c in cand], 1)
        ceil[va] = sc.max(1)
    res["ceil_hardclass_seen_only"] = float(ceil.mean())
    # linearity check: exact 120-perm search == sort by expected gain (use a smoothed kNN-like p)
    print(res, flush=True)

if which in ("all", "knn"):
    from sklearn.decomposition import PCA
    for k in (15, 40, 80):
        Sexp = np.zeros((n, 5)); Smap = np.zeros((n, 5)); Sopt = np.zeros((n, 5))
        for tr, va in F:
            Z = std_fit(W, tr)
            pca = PCA(40, random_state=0).fit(Z[tr]); Ztr, Zva = pca.transform(Z[tr]), pca.transform(Z[va])
            d2 = ((Zva[:, None, :] - Ztr[None]) ** 2).sum(-1)
            nn = np.argsort(d2, 1)[:, :k]
            p = np.zeros((len(va), 120))
            for r in range(len(va)):
                np.add.at(p[r], rc[tr][nn[r]], 1.0 / k)
            Sexp[va] = p @ PG
            Smap[va] = PG[p.argmax(1)]
            Sopt[va] = best_perm_scores(p)
        rep(f"knn{k}_expgain", Sexp); rep(f"knn{k}_mapclass", Smap); rep(f"knn{k}_permsearch", Sopt)
    print(time.time() - t0, flush=True)


class Net(torch.nn.Module):
    def __init__(self, d_in, out, hid=256, p=0.3):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d_in, hid), torch.nn.GELU(), torch.nn.Dropout(p),
                                     torch.nn.Linear(hid, hid), torch.nn.GELU(), torch.nn.Dropout(p),
                                     torch.nn.Linear(hid, out))

    def forward(self, x):
        return self.f(x)


PGt = torch.tensor(PG, dtype=torch.float32)


def pl_nll(s, ordr):
    # Plackett-Luce NLL for full ranking; s (b,5), ordr (b,5) site at each position
    so = torch.gather(s, 1, ordr)
    lse = torch.logcumsumexp(so.flip(1), 1).flip(1)
    return (lse - so)[:, :4].sum(1).mean()


def train_mlp(loss_kind, tr, va, Z, seed, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    out = 120 if loss_kind == "perm" else 5
    net = Net(Z.shape[1], out)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    Xtr = torch.tensor(Z[tr], dtype=torch.float32); Rtr = torch.tensor(R[tr], dtype=torch.float32)
    Otr = torch.tensor(order[tr]); Ctr = torch.tensor(rc[tr])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 64
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr))
        for b in range(0, len(tr), bs):
            idx = perm[b:b + bs]
            xb = Xtr[idx] + 0.05 * torch.randn_like(Xtr[idx])
            s = net(xb)
            if loss_kind == "mse":
                loss = ((s - Rtr[idx]) ** 2).mean()
            elif loss_kind == "pl":
                loss = pl_nll(s, Otr[idx])
            elif loss_kind == "listnet":
                loss = -(torch.softmax(Rtr[idx] * 5, 1) * torch.log_softmax(s, 1)).sum(1).mean()
            elif loss_kind == "perm":
                loss = torch.nn.functional.cross_entropy(s, Ctr[idx], label_smoothing=0.05)
            elif loss_kind == "plmse":
                loss = pl_nll(s[:, :5], Otr[idx]) + 5 * ((torch.tanh(s) - (Rtr[idx] - 0.5)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        o = net(torch.tensor(Z[va], dtype=torch.float32))
    if loss_kind == "perm":
        p = torch.softmax(o, 1).numpy()
        return p @ PG, p
    if loss_kind == "pl":
        # expected gain under PL: enumerate 120 perms
        s = o.numpy()
        lp = np.zeros((len(va), 120))
        for c, pm in enumerate(PERMS):
            so = s[:, pm]
            lse = np.log(np.cumsum(np.exp(so[:, ::-1] - so.max(1, keepdims=True)), 1)[:, ::-1]) + so.max(1, keepdims=True)
            lp[:, c] = (so - lse)[:, :4].sum(1)
        return s, exp_gain_from_logp(lp)
    return o.numpy(), None


if which in ("all", "mlp"):
    kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mse", "pl", "perm", "listnet"]
    seeds = (0, 1)
    for kind in kinds:
        S = np.zeros((n, 5)); S2 = np.zeros((n, 5)); P = np.zeros((n, 120))
        for tr, va in F:
            Z = std_fit(W, tr)
            for sd in seeds:
                a, b = train_mlp(kind, tr, va, Z, sd)
                S[va] += a / len(seeds)
                if kind == "pl":
                    S2[va] += b / len(seeds)
                if kind == "perm":
                    P[va] += b / len(seeds)
        rep(f"mlp_{kind}", S)
        if kind == "pl":
            rep("mlp_pl_expgain", S2)
        if kind == "perm":
            rep("mlp_perm_permsearch", best_perm_scores(P))
            rep("mlp_perm_mapclass", PG[P.argmax(1)])
        print(kind, time.time() - t0, flush=True)
print(res)
