import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, wf_harness as h
import solution as S
torch.set_num_threads(6)
torch.backends.cudnn.benchmark = False
DEV = torch.device("cpu")
EPOCHS, BATCH, LR, TEMP, NOISE, SEEDS = 70, 64, 2e-3, 0.15, 0.10, (0, 1, 2)


class TCN(nn.Module):
    def __init__(self, n_stat, d=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(9, 32, 3, padding=1), nn.GELU(), nn.Conv1d(32, 64, 3, padding=1),
                                  nn.GELU(), nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(128 + n_stat, d), nn.GELU(), nn.Dropout(0.1))
        self.site = nn.Parameter(torch.zeros(5, d))
        self.lead = nn.Linear(4, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True), 2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, seq, stat, lead):
        B = seq.shape[0]
        x = self.conv(seq.reshape(B * 5, 9, 8))
        x = torch.cat([x.mean(-1), x.amax(-1)], -1).reshape(B, 5, -1)
        if stat is not None:
            x = torch.cat([x, stat], -1)
        hgt = self.proj(x) + self.site[None] + self.lead(lead)[:, None]
        return self.head(self.enc(hgt)).squeeze(-1)


class TTX(nn.Module):
    def __init__(self, n_stat, d=64):
        super().__init__()
        self.tok = nn.Linear(9, d)
        self.pos = nn.Parameter(torch.zeros(8, d))
        self.tenc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True), 1)
        self.proj = nn.Sequential(nn.Linear(d + n_stat, d), nn.GELU(), nn.Dropout(0.1))
        self.site = nn.Parameter(torch.zeros(5, d))
        self.lead = nn.Linear(4, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True), 2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, seq, stat, lead):
        B = seq.shape[0]
        t = self.tok(seq.reshape(B * 5, 9, 8).transpose(1, 2)) + self.pos[None]
        t = self.tenc(t).mean(1).reshape(B, 5, -1)
        if stat is not None:
            t = torch.cat([t, stat], -1)
        hgt = self.proj(t) + self.site[None] + self.lead(lead)[:, None]
        return self.head(self.enc(hgt)).squeeze(-1)


def prep(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(Xs, dtype=np.float32)


def fit_predict(cls, use_stats, Xtr, Rtr, ltr, Xte, lte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    str_, ste_ = prep(Xtr), prep(Xte)
    if use_stats:
        _, Ftr = S.site_stats(Xtr); _, Fte = S.site_stats(Xte)
        mu = Ftr.reshape(-1, Ftr.shape[-1]).mean(0); sd = Ftr.reshape(-1, Ftr.shape[-1]).std(0) + 1e-6
        Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
        n_stat = Ftr.shape[-1]
    else:
        Ftr = Fte = None; n_stat = 0
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    sq, st, ld, y = t(str_), (t(Ftr) if use_stats else None), t(S.lead_onehot(ltr)), t(Rtr)
    sq2, st2, ld2 = t(ste_), (t(Fte) if use_stats else None), t(S.lead_onehot(lte))
    tgt = F.softmax(y / TEMP, 1)
    model = cls(n_stat).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = EPOCHS * ((len(sq) + BATCH - 1) // BATCH)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps + 1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(sq), generator=g)
        for i in range(0, len(sq), BATCH):
            b = perm[i:i + BATCH]
            s = model(sq[b] + NOISE * torch.randn_like(sq[b]), (st[b] if use_stats else None), ld[b])
            loss = F.mse_loss(torch.sigmoid(s), y[b]) + 0.1 * -(tgt[b] * F.log_softmax(s, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(sq2, st2, ld2)).numpy()


d = h.load(); R, X, lead, g = d['R'], d['X'], d['lead'], d['groups']
folds = h.folds(g); t1 = R.argmax(1)
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)
V3 = 0.35 * z(np.load('oof/v1__nn.npy')) + 0.45 * z(np.load('oof/an__base_xsite.npy')) + 0.2 * z(np.load('oof/v2__knn_p40k40.npy'))
print('v3 ref OOF', round(float(h.row_scores(V3, R).mean()), 4), 'top1acc', round(float((V3.argmax(1) == t1).mean()), 4), flush=True)
for name, cls, use_stats in [('tcn_stats', TCN, True), ('tcn_pure', TCN, False), ('ttx_stats', TTX, True)]:
    O = np.zeros_like(R)
    for k, (a, b) in enumerate(folds):
        O[b] = np.mean([fit_predict(cls, use_stats, X[a], R[a], lead[a], X[b], lead[b], s) for s in SEEDS], 0)
        print(f'  {name} fold{k} {h.row_scores(O[b], R[b]).mean():.4f}', flush=True)
    np.save(f'oof/tcn__{name}.npy', O)
    print(f'[{name}] OOF {h.row_scores(O, R).mean():.4f} top1acc {(O.argmax(1)==t1).mean():.4f}', flush=True)
    for w in (0.2, 0.3, 0.4):
        Bl = (1 - w) * V3 + w * z(O)
        print(f'   v3 + {w:.1f}*{name} -> OOF {h.row_scores(Bl, R).mean():.4f} top1acc {(Bl.argmax(1)==t1).mean():.4f}', flush=True)
