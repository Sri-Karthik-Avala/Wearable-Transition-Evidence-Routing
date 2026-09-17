# made by - Karthik
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.decomposition import PCA

SITES = ("RWrist", "RUpArm", "Waist", "LThigh", "LAnkle")
GAINS = np.array([1.0, 0.70, 0.45, 0.25, 0.10], dtype=np.float32)
LEADS = (1.0, 3.0, 5.0, 7.0)
SITE_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NN_SEEDS = (0, 1, 2)
NN_EPOCHS = 80
NN_BATCH = 64
NN_LR = 2e-3
NN_TEMP = 0.15
NN_NOISE = 0.10
LGB_ROUNDS = 300
KNN_PCA = 40
KNN_K = 40
BLEND = {"nn": 0.35, "lgb": 0.45, "knn": 0.2}

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_seq(series):
    return np.stack([np.asarray(json.loads(s), dtype=np.float32) for s in series]).reshape(-1, 8, 45)


def relevance(preds):
    R = np.zeros((len(preds), 5), dtype=np.float32)
    for i, p in enumerate(preds):
        for r, s in enumerate(str(p).split(">")):
            R[i, SITES.index(s)] = GAINS[r]
    return R


def site_stats(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)
    norms = np.stack([np.linalg.norm(Xs[..., k:k + 3], axis=-1) for k in (0, 3, 6)], -1)
    parts = [
        Xs.std(2),
        Xs.mean(2),
        Xs[:, :, -1] - Xs[:, :, 0],
        np.abs(np.diff(Xs, axis=2)).mean(2),
        Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2),
        norms.std(2),
        norms.mean(2),
        np.abs(np.diff(norms, axis=2)).mean(2),
    ]
    return Xs, np.concatenate(parts, -1).astype(np.float32)


def lead_onehot(lead):
    return np.stack([(lead == l).astype(np.float32) for l in LEADS], 1)


class SiteRanker(nn.Module):
    def __init__(self, n_stat, d=64):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(72, d), nn.GELU(), nn.Dropout(0.1))
        self.stat = nn.Sequential(nn.Linear(n_stat, d), nn.GELU(), nn.Dropout(0.1))
        self.site = nn.Parameter(torch.zeros(5, d))
        self.lead = nn.Linear(4, d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, seq, stat, lead):
        h = self.seq(seq) + self.stat(stat) + self.site[None] + self.lead(lead)[:, None]
        return self.head(self.enc(h)).squeeze(-1)


def fit_predict_nn(Xtr, Rtr, ltr, Xte, lte, seed):
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
    target = F.softmax(y / NN_TEMP, 1)
    model = SiteRanker(Ftr.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=1e-2)
    steps = NN_EPOCHS * ((len(Str) + NN_BATCH - 1) // NN_BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=NN_LR, total_steps=steps + 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(NN_EPOCHS):
        model.train()
        perm = torch.randperm(len(Str), generator=g).to(DEVICE)
        for i in range(0, len(Str), NN_BATCH):
            b = perm[i:i + NN_BATCH]
            sq = seq_tr[b] + NN_NOISE * torch.randn_like(seq_tr[b])
            s = model(sq, st_tr[b], ld_tr[b])
            loss = -(target[b] * F.log_softmax(s, 1)).sum(1).mean() + F.mse_loss(torch.sigmoid(s), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        return torch.log_softmax(model(seq_te, st_te, ld_te), 1).cpu().numpy()


def flat_features(X, l):
    _, Fs = site_stats(X)
    n = len(Fs)
    rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
    return np.concatenate([Fs.reshape(n, -1), rel.reshape(n, -1), X[:, -1], l[:, None]], 1).astype(np.float32)


def fit_predict_knn(Xtr, Rtr, ltr, Xte, lte):
    A, B = flat_features(Xtr, ltr), flat_features(Xte, lte)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    pca = PCA(KNN_PCA, random_state=0).fit((A - mu) / sd)
    Za, Zb = pca.transform((A - mu) / sd), pca.transform((B - mu) / sd)
    out = np.zeros((len(B), 5), dtype=np.float32)
    for i in range(0, len(Zb), 256):
        d2 = ((Zb[i:i + 256, None, :] - Za[None]) ** 2).sum(-1)
        nn_idx = np.argsort(d2, 1, kind="stable")[:, :KNN_K]
        out[i:i + 256] = Rtr[nn_idx].mean(1)
    return out


def cross_site_features(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9)
    out = []
    for k in (0, 3, 6):
        v = Xs[..., k:k + 3]
        nrm = np.linalg.norm(v, axis=-1)
        nc = nrm - nrm.mean(1, keepdims=True)
        mv = v.mean(1)
        mvn = mv / (np.linalg.norm(mv, axis=-1, keepdims=True) + 1e-6)
        dv = v[:, -1] - v[:, 0]
        for i, j in SITE_PAIRS:
            out.append((nc[:, :, i] * nc[:, :, j]).mean(1) / (nc[:, :, i].std(1) * nc[:, :, j].std(1) + 1e-6))
            out.append((mvn[:, i] * mvn[:, j]).sum(-1))
            out.append(np.linalg.norm(dv[:, i], axis=-1) - np.linalg.norm(dv[:, j], axis=-1))
    sd = Xs.std(1).mean(-1)
    ranks = np.argsort(np.argsort(sd, 1), 1).astype(np.float32)
    return np.concatenate([np.column_stack(out), ranks], 1).astype(np.float32)


def fit_predict_lgb(Xtr, Rtr, ltr, Xte, lte):
    A = np.concatenate([flat_features(Xtr, ltr), cross_site_features(Xtr)], 1)
    B = np.concatenate([flat_features(Xte, lte), cross_site_features(Xte)], 1)
    out = np.zeros((len(B), 5), dtype=np.float32)
    for s in range(5):
        m = lgb.LGBMRegressor(n_estimators=LGB_ROUNDS, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=s,
                              n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(A, Rtr[:, s])
        out[:, s] = m.predict(B)
    return out


def row_z(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)


def fit_predict(Xtr, Rtr, ltr, Xte, lte):
    members = {
        "nn": np.mean([fit_predict_nn(Xtr, Rtr, ltr, Xte, lte, s) for s in NN_SEEDS], 0),
        "lgb": fit_predict_lgb(Xtr, Rtr, ltr, Xte, lte),
        "knn": fit_predict_knn(Xtr, Rtr, ltr, Xte, lte),
    }
    return sum(w * row_z(members[k]) for k, w in BLEND.items()), members


def to_strings(S):
    order = np.argsort(-S, 1, kind="stable")
    return [">".join(SITES[j] for j in o) for o in order]


if __name__ == "__main__":
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    Xtr, Xte = parse_seq(train.sensor_sequence), parse_seq(test.sensor_sequence)
    Rtr = relevance(train.prediction.values)
    ltr = train.lead_offset_s.values.astype(np.float32)
    lte = test.lead_offset_s.values.astype(np.float32)
    S, _ = fit_predict(Xtr, Rtr, ltr, Xte, lte)
    submission = pd.DataFrame({"id": test.id.values, "prediction": to_strings(S)})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    print("wrote", len(submission), "rows")
