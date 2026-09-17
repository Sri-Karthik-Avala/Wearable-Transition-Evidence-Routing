# Wearable Transition Evidence Routing — Master Experiment Report

Written for an external reviewer (another model) to find missed improvement routes. Everything below is measured; every OOF number uses the same 5-fold participant-grouped split unless marked. Where a number is partial (fewer folds, a failed or overwritten run), it says so.

**Bottom line:** three leaderboard submissions landed at 0.7563, 0.7569 and 0.7543. Local 5-fold OOF for all three is 0.767–0.769. The leaderboard standard error at 17 test participants is about ±0.017, so the three are statistically identical. The target to beat is 0.80. The loss is almost entirely in picking the **top-1 site**: accuracy is 0.549, and the oracle top-1 with model order for the rest would score 0.924.

---

## 1. Problem

### 1.1 Input
- Each row is one window: 8 ordered frames × 45 channels, flattened into `sensor_sequence` (360 floats, reshape `(8, 45)` row-major).
- The 45 channels are 5 sites × 9 channels, in fixed order: RWrist, RUpArm, Waist, LThigh, LAnkle. Each site has acc x,y,z, gyro x,y,z, mag x,y,z.
- The 4-second context ends `lead_offset_s` ∈ {1, 3, 5, 7} seconds before an annotated activity boundary.
- Values are standardized channel-wise using the train rows only.

### 1.2 Target
A full ranking of the 5 sites, e.g. `RWrist>LThigh>LAnkle>RUpArm>Waist`. The organizer computes it from real signal that is never in the input:
- Take 13 raw samples just before the boundary and the first 13 raw samples after it.
- Per site, the evidence is the mean over its 9 channels of `|post_mean − pre_mean| / (pre_std + 0.05)`, in **raw** units.
- Sites are sorted by evidence, highest first, with ties broken by the fixed site order.

### 1.3 Metric
Graded NDCG@5 with position gains (1.0, 0.70, 0.45, 0.25, 0.10). It is rescaled so the reverse ranking scores 0 and the ideal ranking scores 1. The score is the unweighted mean over all test rows.
- The metric is linear in per-site gain given the predicted order. So **sorting sites by expected gain E[gain_s | x] is the Bayes-optimal decode.** This was verified empirically: an exact search over all 120 permutations gives the same score as sorting by expected gain.

### 1.4 Data facts (measured)

| Quantity | Value |
|---|---|
| Train rows | 2104 = 526 boundaries × 4 lead offsets. All 4 rows of a boundary share the **same** target. |
| Test rows | 592 = 148 boundaries × 4 lead offsets, 17 participants disjoint from train |
| Participant groups (`validation_groups.csv`) | 59 groups of ~36 rows each (≈9 boundaries per participant) |
| Unique target rankings in train | 107 of 120 possible |
| Top-1 site frequency (train) | RWrist 672, LThigh 460, LAnkle 424, RUpArm 328, Waist 220 |
| Mean relevance per site | RWrist 0.636, RUpArm 0.492, LThigh 0.473, LAnkle 0.470, Waist 0.428 |
| Train channel stats | mean 0, std 1 |
| Test channel stats | mean −0.068, std 1.19 |
| Random ranking score | 0.468 |
| Prior ranking (sort by mean train relevance) | 0.592 OOF |

### 1.5 Platform constraints (Project Eris), which any proposal must respect
- One end-to-end `solution.py <public_dir> <submission_csv>`. It must contain **real training** inside the script; an inference-only or rule-based solution is rejected.
- No hardcoded dataset findings. Architecture choices are fine; tuned thresholds must be derived at runtime, and HPO inside the script is allowed.
- No `os.environ`, no subprocess or pip, no filesystem walks, no wall-clock-adaptive logic, and `cudnn.benchmark` must be False.
- **Test rows must be predicted one at a time.** Nothing may be fitted on test-set statistics: no cross-test normalization, no pseudo-labels, no matching test rows to each other.
- Do not reverse-map ids and do not use `validation_group` as a feature.
- Pretrained weights from Hugging Face or timm are allowed; GitHub code and `torch.hub` are not.
- Grader hardware: GPU challenge, 1 h budget. Our full run takes about 3.5 min.
- Each leaderboard score costs a credit, and a hidden private leaderboard decides the final rank.

---

## 2. Validation setup
- `wf_harness.py`: `GroupKFold(5)` on `validation_group`, an exact metric re-implementation, OOF matrices saved as `oof/<name>.npy` with shape (2104, 5), and a boundary bootstrap for leaderboard noise.
- **Leaderboard noise:** bootstrapping 17 participant groups from the v1 OOF gives a standard error of **0.017–0.018**.
- **Local CV tracks the leaderboard.**
  - v1: OOF 0.7668, LB 0.7563.
  - v2: OOF 0.7684, LB 0.7569. The predicted +0.0016 appeared on the leaderboard as +0.0006.
  - v3: OOF 0.7689, LB 0.7543.
- **Per-fold spread is large:** v1 folds scored 0.766 / 0.790 / 0.778 / 0.731 / 0.769. Any 2-fold number is unreliable. Early in the session a 2-fold 0.778 looked like a gain and wasn't.
- **Train-vs-test shift:** adversarial validation AUC is 0.64–0.91 on raw feature groups, but only **0.00–0.03 above a pseudo-baseline** that splits train participants 17-vs-rest. So the test is "new participants", not a different distribution. Level-only features carry most of the participant identity.

---

## 3. Submitted solutions

| Version | What it is | 5-fold OOF | Leaderboard |
|---|---|---|---|
| v1 | Site-token transformer (3 seeds) + per-site LightGBM, row-z-scored 50/50 | 0.7668 | **0.7563** |
| v2 | v1 + kNN expected-gain member, weights NN 0.4 / LGB 0.4 / kNN 0.2 | 0.7684 | **0.7569** |
| v3 | v2 with cross-site features added to LightGBM, weights 0.35 / 0.45 / 0.2 | 0.7689 | **0.7543** |

v3's `solution.py` source is appended at the end of this report.

### 3.1 Shared per-site features (`site_stats`)
- Reshape to (n, 5 sites, 8 frames, 9 channels).
- Per site, 54 stats:
  - per-channel std, mean, last − first, mean |Δ|, and (mean of last 2 frames − mean of first 6);
  - for the acc, gyro and mag vector norms: std, mean, and mean |Δ| of the norm.

### 3.2 Member A: site-token transformer (`SiteRanker`)
- **Tokens:** 5 tokens, one per site. Each token = Linear(8×9 raw sequence → 64) + Linear(54 standardized stats → 64) + a learned site embedding + Linear(lead one-hot → 64).
- **Encoder:** 2-layer pre-norm TransformerEncoder (d = 64, 4 heads, FF 128, dropout 0.1) → LayerNorm → Linear head giving one score per site.
- **Loss:** ListNet cross-entropy against `softmax(gain / 0.15)` + MSE(sigmoid(score), gain).
- **Training:** AdamW (lr 2e-3, wd 1e-2), OneCycle schedule, 80 epochs, batch 64, Gaussian input noise σ = 0.1, 3 seeds averaged.
- **Output:** log-softmax of the scores.
- **5-fold OOF alone: 0.7448.**

### 3.3 Member B: per-site LightGBM regression
- **Features:** the 5×54 stats, the same stats divided by their 5-site mean (270), the raw last frame (45), and the lead.
- **Models:** 5 independent LGBMRegressors, one per site, each regressing that site's gain. n_estimators 300, lr 0.03, num_leaves 15, min_child_samples 20, subsample 0.8, colsample 0.5.
- **5-fold OOF: 0.7599.** Another explorer's equivalent reference scored 0.7577.
- **v3 adds 95 cross-site features:**
  - for each sensor type × each of the 10 site pairs: the correlation of the two sites' centred vector-norm series over 8 frames;
  - the cosine between the two sites' mean vectors;
  - the difference of their |last − first| norms;
  - plus each site's rank of within-window std.
- **LightGBM alone with cross-site features: 0.7640, up from 0.7599 (+0.0041).** This is the only feature change that clearly helped a single model. **Inside the blend it added nothing.**

### 3.4 Member C: kNN expected gain
- Standardize member B's flat features (statistics fitted on train), reduce with PCA(40), then Euclidean 40-NN.
- Score = mean of the neighbours' gain vectors, which equals Σ p(ranking) · gains.
- **5-fold OOF: 0.7480.** With PCA 32 / k 25: 0.7464.

### 3.5 Blending
- Each member's scores are z-scored within the row, then combined as a weighted sum, and sites are sorted.
- v1 weight curve (w on the NN, remainder on LightGBM), 5-fold:

| w_nn | 0 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|
| OOF | 0.7599 | 0.7663 | 0.7668 | 0.7575 | 0.7448 |

---

## 4. Everything else tried (5-fold OOF unless marked)

### 4.1 Tabular / feature engineering (LightGBM per-site unless noted)

| Variant | OOF | Note |
|---|---|---|
| Baseline wide stats (per-channel std, mean, last − first, mean \|Δ\|, last frame, late − early) | 0.7580 | first recon |
| + per-site summaries (channel-mean of std, \|Δ\|, \|last − mean\|; raw, ÷5-site mean, rank among 5) | 0.7608 | small gain |
| Long format (one row per window × site, site categorical, own + all sites' features) | 0.7614 | ≈ same |
| Long-format LambdaRank | 0.7569 | no better than regression |
| + vector norms (mean, std, last, \|Δ\|, min, max) + per-channel min/max | 0.7524 | hurt |
| + inverse "quietness" features 1/(std + 0.05), 1/(last diff + 0.05) | 0.7510 | hurt, although the target divides by pre-std |
| v1 LightGBM features (section 3.3) | 0.7599 | |
| + cross-site features (section 3.3) | **0.7640** | best single model |
| LightGBM, 3-seed bagging | +0.004 (2-fold only) | not confirmed on 5 folds |
| kNN on site-relative features, k = 100 / 30 | 0.7287 / 0.7278 | |
| Ridge on relative features, α = 100 / 10 | 0.7165 / 0.7088 | |
| CatBoost MultiRMSE | — | failed: host out of memory, never run |

### 4.2 Ranking-structure models (explorer "tgt")

| Variant | OOF |
|---|---|
| Pairwise LightGBM (10 pair classifiers) → expected gain, T = 0.5 | 0.7544 |
| Pairwise → Borda count | 0.7540 |
| kNN over ranking classes, PCA32, k = 25, expected-gain decode | 0.7482 |
| Same with exact 120-permutation expected-NDCG search | identical to expected-gain decode |
| Same with MAP decode (single most likely ranking) | 0.69–0.705 (much worse) |
| Plackett–Luce listwise MLP | 0.7363 |
| MLP with MSE on gains | 0.7340 |
| LightGBM multiclass over the top-1 site (5 classes) | 0.7346 alone; top-1 accuracy 0.541 vs 0.549 for regression |
| Multinomial logit over rankings | failed: MemoryError |

### 4.3 Neural variants

| Variant | OOF | Note |
|---|---|---|
| v1 transformer alone | 0.7448 | 5-fold |
| Transformer with the ListNet term at 0.1, MSE on sigmoid(gain) primary, sigmoid output ("v2 NN") | 0.7434 / 0.7779 on folds 0 / 1 | vs v1 0.7392 / 0.7686; the blend with 3-seed LightGBM did not improve (2-fold 0.7783 vs 0.7777) |
| Transformer re-runs by the explorer | fold-level only: 0.7357 (fold 1, 3 seeds), 0.7298, 0.7262, 0.7534 | runs killed by out-of-memory and by closed windows; saved `stx__v1.npy` 0.725 is suspect |
| Temporal per-site Conv1d + cross-site attention (explorer "tdl") | fold 0: 0.7302 only | **never completed**: host out of memory, CPU starved, ~26 min per config |
| Temporal CNN over all 45 channels / GRU-attention | — | never completed |

### 4.4 Ensembles

| Blend | 5-fold OOF |
|---|---|
| v1 (NN 0.5 + LGB 0.5) | 0.7668 |
| v1 + kNN 0.2 (= v2) | 0.7684 |
| v1 + kNN 0.1 + top-1 multiclass 0.1 | 0.7676 |
| v3 LightGBM + NN 0.5 / 0.5 (no kNN) | 0.7660 |
| v3 + kNN 0.4 / 0.4 / 0.2 | 0.7680 |
| v3 + kNN 0.35 / 0.45 / 0.2 (= v3) | 0.7689 |
| LightGBM reference + kNN (0.5 / 0.5), no NN | 0.7631 |
| LightGBM reference + top-1 0.2 + kNN 0.5, no NN | 0.7665 |
| **Nested greedy blend over 10 saved OOFs** (weights picked on 4 folds, scored on the 5th) | **0.7679** |
| Same greedy blend, in-sample (inflated) | 0.7733. Weights: v1 NN 0.17, v1 LGB 0.42, kNN(pca32, k25) 0.33, PL-MLP 0.08 |

**Conclusion:** every model family and blend lands at 0.766–0.769 OOF. kNN and the NN supply the same kind of diversity; once one is in the blend the other adds about 0.

---

## 5. Diagnostics: where the loss is

### 5.1 Top-1 site is the loss
Measured on the per-site LightGBM OOF:
- Top-1 accuracy: **0.549**. It is flat across leads: 0.553 / 0.559 / 0.538 / 0.548.
- Oracle top-1 + model order for positions 2–5: **0.924**.
- Model top-1 + oracle order for positions 2–5: **0.849**.

### 5.2 Top-1 confusion (rows = true, columns = predicted), per-site LightGBM OOF

| true \ pred | LAnkle | LThigh | RUpArm | RWrist | Waist |
|---|---|---|---|---|---|
| LAnkle | **242** | 48 | 21 | 91 | 22 |
| LThigh | 90 | **238** | 36 | 59 | 37 |
| RUpArm | 13 | 13 | **72** | 169 | 61 |
| RWrist | 64 | 16 | 58 | **511** | 23 |
| Waist | 27 | 29 | 28 | 43 | **93** |

- RUpArm is mostly predicted as RWrist (169 vs 72 correct), so the same-limb pair is confused.
- LAnkle and LThigh swap often: 90 and 48. That is the same-leg pair.
- Waist is rarely top-1 and poorly recalled.

### 5.3 Signals checked
- **Lead offset:** the score is flat across leads 1, 3, 5, 7. Windows ending 1 s before the boundary are no more informative than windows ending 7 s before. The signal is the ongoing activity or posture context, not an imminent-change cue.
- **Window quietness:** a site's within-window std (acc, gyro, mag or all) has Spearman ≈ 0 with its true gain (−0.03 to +0.02). "Quietest site is top-1" is right 20–25% of the time, barely above the 20% base rate. So the target's pre-std denominator is not recoverable from the window.
- **Distribution shift:** section 2. No real shift beyond "new participants".

---

## 6. Process problems (so they aren't repeated)
- **Memory:** the grading-style runs shared one machine with another challenge's job, which used 22 GB of commit memory and 100% of the GPU. About half of the neural explorer runs died from lack of memory, so the temporal CNN/GRU family is **untested, not refuted**.
- **Duplicate agents:** messaging running workflow agents started duplicate agent copies, which overwrote each other's OOF files. `stx__v1.npy` and some `tdl__*` files are unreliable.
- **Pop-up windows:** detached launches opened console windows. Closing them killed runs.

---

## 7. Untested or under-tested directions (the reviewer's input is wanted here)

Ordered by my estimate of upside. None of these has been run to completion.

1. **Temporal deep models.** Never completed because of the resource problems in section 6. Worth a clean run on a free GPU:
   - a 1D CNN, TCN or GRU over the 8 frames;
   - a per-site temporal encoder with cross-site attention;
   - multi-task heads: gain regression + top-1 classification + pairwise;
   - 5 or more seeds, with mixup, site-channel dropout and scale jitter to match test std 1.19.
2. **Dedicated confusable-pair models.** Binary models for RUpArm-vs-RWrist and LAnkle-vs-LThigh as re-rankers of the top-2. The confusion in section 5.2 is concentrated there. Needs features that separate the limb segments: relative orientation of wrist vs upper arm, and the gyro ratio between them.
3. **Ceiling estimate.** How much of the top-1 error is irreducible near-ties in the organizer's evidence? We only see ranks, not evidence values. A reviewer might propose a way to estimate the Bayes ceiling, for example agreement between nearest-neighbour boundaries from different participants.
4. **Pretrained time-series encoders** as feature extractors or fine-tuned (Hugging Face models such as MOMENT or Chronos-style encoders). Allowed by the rules. The description says the task is "calibrated for lightweight or medium learned models" and is "not a from-scratch pretraining task".
5. **HPO inside the script** (allowed and encouraged). LightGBM was never tuned beyond defaults, and the transformer was never tuned.
6. **Better target modelling.** Plackett–Luce with a stronger backbone. A latent-evidence regression that treats the ranking as censored ordinal data and learns a per-site evidence score with a pairwise likelihood. Soft targets.
7. **Physics of the target.** Evidence uses raw units with a +0.05 floor, so channel types (acc, gyro, mag) contribute on different scales. Standardization hides the raw scale. Could per-channel-type sub-models, or a learned per-channel weighting inside the model, capture which channel type dominates each site's evidence?
8. **Participant-invariant inputs.** The shift explorer's sweep of per-row centring, dropping magnetometer levels and centring with rank-normalization never reported. Magnetometer absolute levels are likely environment-specific.
9. **Transition-type structure.** Cluster the 107 target rankings, or the pre-boundary posture, into transition archetypes such as sit→stand. Classify the archetype, then rank within it: a hierarchical mixture of experts.
10. **Training-set boundary reconstruction.** The 4 lead windows of a boundary overlap: lead-1 covers [−5, −1] s and lead-3 covers [−7, −3] s. Chaining them *in train only* could give a 10 s history and more training crops. **Compliance risk:** the rules forbid inferring boundary order and using signal outside a row's own sequence. We did not do it, and it should be cleared with a reviewer first.

---

## 8. Files
- `solution.py` (= v3) and `working/solution_v{1,2,3}.py` + `working/submission_v{1,2,3}.csv`: the submitted pairs.
- `wf_harness.py`: shared split, metric and OOF saver. `oof/*.npy`: every saved OOF matrix (see the reliability caveats in section 6).
- Dev scripts:
  - `dev_cv.py`: v1 CV.
  - `dev_members_v2.py`: kNN and top-1 members.
  - `dev_top1.py`: top-1 multiclass test.
  - `dev_xsite.py`: cross-site features.
  - `dev_sol_v2.py`: the v2 NN-head variant.
  - `dev_wf_*`: explorer scripts, some overwritten by duplicate agents.

---

## Appendix A: v3 `solution.py` (as submitted, LB 0.7543)

```python
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
```

---

## 9. Round 2: results of the reviewer's proposed plan (tests 1–3)

All on the same 5-fold split. **Every one of the reviewer's top-3 bets came back negative.** The key reason is in the first row of the last table: the *blend* already picks the top-1 site better than any specialist pipeline built on LightGBM alone.

### 9.1 Test 3 — anatomical relative-motion + per-sensor-type temporal features (246 new features)
Per site × sensor type (acc/gyro/mag): norm mean/std/last−first/mean|Δ|, delta energy, burstiness (max|Δ|/mean|Δ|), norm slope and residual std, mean and min cosine between consecutive vectors. Per pair (RWrist–RUpArm, LThigh–LAnkle, Waist–RWrist, Waist–LThigh) × sensor type: difference-vector norm mean/std/last−first/mean|Δ|, delta energy, cosine of mean vectors, correlation of centred norms, slope.

| Feature set | OOF | Top-1 acc |
|---|---|---|
| base + cross-site | **0.7640** | **0.5646** |
| base + cross-site + anatomical | 0.7614 | 0.5504 |

**Refuted.** Relative limb-segment motion, as constructed here, adds noise. This is the third time a large feature block has hurt (see section 4.1).

### 9.2 Test 1 — dedicated top-1 classifier
LightGBM 5-class multiclass on the base+cross-site features, predicting `argmax(gain)`.

| Model | Top-1 acc | OOF (decoding by its own scores) |
|---|---|---|
| Gain regression (per-site) | **0.5646** | 0.7640 |
| Dedicated top-1 classifier | 0.5399 | 0.7363 |

Using it only to choose the first site, keeping gain order for positions 2–5:

| First site = (1−w)·gain + w·top-1 | w=0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|---|---|
| OOF | 0.7643 | 0.7641 | 0.7619 | 0.7597 | 0.7560 | 0.7537 |
| Top-1 acc | 0.5642 | 0.5632 | 0.5589 | 0.5556 | 0.5494 | 0.5456 |

**Refuted.** A classifier trained directly on the top-1 label is *worse* at the top-1 decision than regressing all five gains. The reviewer's target was >0.58 top-1; the best achieved anywhere in round 2 was 0.5756, by the existing blend. Plausible reason: the gain targets carry the full ordering as supervision (5 graded values per row), while the top-1 label discards it; with only 526 independent boundaries, the classifier is supervision-starved.

### 9.3 Test 2 — conditional pair rerankers (RWrist↔RUpArm, LThigh↔LAnkle)
Binary LightGBM per pair, target `gain_i > gain_j`, trained on all rows with sample weight 1 + 2·[pair is truly top-2], applied only where that pair is the predicted top-2.

On the LightGBM-only pipeline:

| Pair | Rows | Top-1 acc on those rows | OOF |
|---|---|---|---|
| RWrist/RUpArm | 784 | 0.589 → 0.570 | 0.7628 |
| LThigh/LAnkle | 377 | 0.637 → **0.653** | 0.7647 |

On the actual v3 blend:

| Variant | OOF |
|---|---|
| v3 blend unchanged | **0.7689** (top-1 acc 0.5756) |
| + leg reranker (all 354 top-2 rows) | 0.7661 |
| + arm reranker (all 780 top-2 rows) | 0.7664 |
| + leg reranker, margin < 0.2 (34 rows) | 0.7687 |
| + leg reranker, margin < 0.4 (87 rows) | 0.7684 |
| + leg reranker, nested per-fold threshold | 0.7685 (4 of 5 folds chose "do nothing") |

**Refuted on the blend.** The leg reranker's apparent gain exists only against the weaker LightGBM-only ordering. Margin gating does not rescue it: the nested threshold search chooses to disable it.

### 9.4 The finding that explains all three
Top-1 accuracy by pipeline: old per-site LightGBM **0.549** → +cross-site features **0.5646** → **v3 blend 0.5756**.

The ensemble is already the best top-1 discriminator available. Every specialist in round 2 was built on the LightGBM-only path and therefore started 1.1 points of top-1 accuracy behind; replacing the blend's choices with theirs overwrote more correct answers than it fixed. **Any future top-1 work must be benchmarked against the blend's 0.5756, not against LightGBM's 0.5646.**

### 9.5 What round 2 leaves open
Untested or unfinished, in the order I would now try them:
1. **Temporal CNN/TCN** (reviewer's test 6) — still never completed; needs an uncontended GPU. This is the only untried *model family*.
2. **Participant-invariant normalization and the magnetometer ablation** (ALL vs NO-MAG vs MAG-DELTA-ONLY) — the sweep was killed before reporting; cheap and worth finishing.
3. **Ordinal / latent-evidence supervision** (P(rank ≤ k) per site, reconstruct expected gain) — only the Plackett–Luce MLP variant was tried (0.7363) and it used a weak backbone.
4. **HPO** for LightGBM and the transformer — still untested.
5. **Multi-task neural model** (gain + top-1 + pairwise heads on a shared temporal encoder) — combines items 1 and 3, and would put the top-1 head on the strong path rather than the weak one.

Note for whoever picks this up: expected-gain decoding is provably optimal for this metric, so improvements must come from better *estimates* of per-site gain, not from a cleverer decode.

---

## 10. Round 3: ceiling analysis, sensor ablation, invariance

### 10.1 The exchange rate — what 0.80 actually costs
Measured on the v3 blend OOF:
- Oracle top-1 + blend's order for positions 2–5: **0.9239**. Current blend: **0.7689**, top-1 accuracy **0.5756**.
- Therefore **+0.01 top-1 accuracy ≈ +0.0037 NDCG.**
- **Reaching 0.80 needs top-1 accuracy ≈ 0.661**, up from 0.5756 (+8.5 points, ~15% relative).
- Milestones: OOF 0.772 needs top-1 ≈ 0.584; 0.775 ≈ 0.592; 0.780 ≈ 0.606.

Use this to price any proposal before running it: a change that cannot plausibly move top-1 by several points cannot move the score out of noise.

### 10.2 Cross-participant nearest-neighbour agreement (evidence on the ceiling)
PCA-32 over the base+cross-site features; for each row, neighbours are restricted to **different participant groups**.

| k | mean neighbour agreement on true top-1 | majority-vote top-1 acc |
|---|---|---|
| 1 | 0.468 | 0.468 |
| 5 | 0.456 | 0.497 |
| 10 | 0.455 | 0.530 |
| 20 | 0.450 | **0.548** |
| 50 | 0.440 | 0.533 |

**The model (0.5756) already beats every kNN transfer vote.** Feature-space similarity across participants carries less signal than the fitted models extract, so there is no easy unused structure; the remaining gain must come from a better function, not a better lookup.

### 10.3 Error concentration by the blend's own confidence margin

| Margin quintile | n | top-1 acc | NDCG |
|---|---|---|---|
| 0.00–0.33 | 421 | 0.375 | 0.7011 |
| 0.33–0.63 | 421 | 0.492 | 0.7450 |
| 0.63–0.98 | 420 | 0.607 | 0.7878 |
| 0.98–1.47 | 421 | 0.651 | 0.7831 |
| 1.47–2.34 | 421 | 0.753 | 0.8274 |

The margin is informative, so the model does know when it is unsure. But note round 2: a specialist applied to the low-margin rows still lost, because it was less accurate there than the blend.

### 10.4 Per-site recall, and why RUpArm's 0.21 is correct behaviour

| True top-1 site | n | blend recall |
|---|---|---|
| RWrist | 672 | 0.777 |
| LAnkle | 424 | 0.630 |
| LThigh | 460 | 0.567 |
| Waist | 220 | 0.418 |
| RUpArm | 328 | **0.210** |

Mean true gain of the *substituted* site, on rows where each site is truly top-1:
- True RUpArm → **RWrist is 2nd in 63% of rows, mean gain 0.566**.
- True LThigh → LAnkle 2nd in 33%, mean gain 0.435. True LAnkle → LThigh 2nd in 25%, mean gain 0.367.
- True Waist → RWrist 2nd in 20%, mean gain 0.378.

**Do not class-balance to raise RUpArm recall.** Under expected-gain decoding, predicting RWrist on an RUpArm row is a cheap error (0.566 vs 1.0), and the decode is already making the right bet. Forcing balance trades cheap errors for expensive ones.

### 10.5 Leave-one-sensor-type-out ablation
Single shared feature builder (per-site per-channel std, mean, last−first, mean |Δ|, late−early, plus norm std/mean/mean|Δ|), so rows are directly comparable to each other but not to the richer 0.7640 set.

| Sensors | dims | OOF | top-1 acc |
|---|---|---|---|
| ALL | 541 | **0.7615** | 0.5580 |
| GYRO+MAG | 361 | 0.7578 | 0.5423 |
| ACC+GYRO | 361 | 0.7526 | 0.5456 |
| ACC+MAG | 361 | 0.7497 | 0.5461 |
| MAG only | 181 | 0.7502 | 0.5456 |
| ACC only | 181 | 0.7456 | 0.5399 |
| GYRO only | 181 | 0.7330 | 0.4976 |

**The magnetometer is the strongest single sensor, not a liability**, and every subset loses to ALL. The reviewer's hypothesis that MAG might be actively hurting is refuted.

### 10.6 Participant-invariant normalization

| Variant | OOF | top-1 acc |
|---|---|---|
| ALL, raw standardized levels | **0.7615** | 0.5580 |
| magnetometer level removed (MAG temporally centred) | 0.7596 | 0.5547 |
| per-channel temporal z-score | 0.7580 | 0.5442 |
| per-channel temporal centring (levels removed) | 0.7548 | 0.5480 |

**Every invariance transform loses.** Absolute levels encode gravity direction, i.e. posture, which is genuinely predictive; the participant-identity signal rides along with real information. Combined with section 2 (adversarial AUC only 0.00–0.03 above the participant pseudo-baseline), the "distribution shift" route is closed.

### 10.7 Status after round 3
Refuted or closed: anatomical relative features, standalone top-1 classifier, pair rerankers, sensor ablation, participant-invariant normalization, class-balancing RUpArm, kNN-style lookup, generic blending.

Still open, in priority order: **temporal models (TCN and a temporal-token transformer — running at the time of writing)**, ordinal / latent-evidence multi-task supervision on a strong backbone, pretrained time-series embeddings as features, LightGBM HPO on the best representation, and a site-pair *magnitude* regression (predict gain differences, not just the sign) used as a correction to the base gains.

---

## 11. Round 4: temporal models (the reviewer's #1 and #2) — refuted

Small per-site temporal encoders, 5-fold, 3 seeds averaged, CPU. Loss MSE(sigmoid(score), gain) + 0.1·ListNet, AdamW 2e-3, OneCycle, 70 epochs, batch 64, input noise 0.1.
- **tcn_stats**: per-site Conv1d(9→32→64→64, k=3) + mean/max pool, concatenated with the 54 handcrafted stats, + site embedding + lead, then 2-layer cross-site TransformerEncoder (d=64).
- **tcn_pure**: identical without the handcrafted stats (pure temporal path, for error diversity).
- **ttx_stats**: 8 temporal tokens per site, Linear(9→64) + positional embedding → 1-layer temporal TransformerEncoder → mean pool, then the same cross-site stack. This is the architecture fix the reviewer asked for: the shipped transformer flattens each site's 8×9 sequence through a single Linear(72→64).

| Model | OOF | top-1 acc |
|---|---|---|
| **tcn_stats** | 0.7339 | 0.5242 |
| **ttx_stats** | 0.7307 | 0.5304 |
| **tcn_pure** | 0.7205 | 0.4986 |
| (reference) shipped transformer | 0.7448 | — |
| (reference) LightGBM + cross-site | 0.7640 | 0.5646 |
| (reference) v3 blend | **0.7689** | **0.5756** |

Blended into v3:

| Weight | tcn_stats | tcn_pure | ttx_stats |
|---|---|---|---|
| 0.2 | 0.7681 | **0.7698** | 0.7688 |
| 0.3 | 0.7657 | 0.7661 | 0.7653 |
| 0.4 | 0.7636 | 0.7611 | 0.7616 |

Best case is v3 + 0.2·tcn_pure = **0.7698, i.e. +0.0009 over v3** — noise, and the weight was chosen on the same rows it is scored on. Top-1 reaches 0.5813 at best, below the 0.584 needed for 0.772.

**Interpretation.** Explicit temporal modelling does not beat summary statistics here, and the ordering is consistent: the more temporal structure a model is forced to use (tcn_pure, no stats) the worse it gets. Two plausible reasons: (a) only 526 independent boundaries, so a temporal encoder has far more capacity than supervision; (b) 8 frames over 4 s is a coarse snapshot — combined with the round-3 finding that the score is flat across lead offsets and that window quietness has zero rank correlation, the input behaves as a **posture/activity snapshot, not a trajectory**. Summary statistics already capture nearly all of it.

### 11.1 Overall conclusion after four rounds
Every model family tried — gradient boosting, transformers, kNN, Plackett–Luce, pairwise, ordinal-ish decodes, temporal CNN/TCN — lands between 0.72 and 0.7640 alone, and every blend lands at 0.766–0.770. The v3 blend at 0.7689 / top-1 0.5756 is, on this evidence, the plateau of this input representation.

**Reaching 0.80 requires top-1 accuracy ≈ 0.661 (section 10.1).** No experiment in four rounds moved top-1 beyond 0.5813, and cross-participant kNN transfer tops out at 0.548. A reviewer proposing further work should first say *by what mechanism* their idea adds ~8 points of top-1 accuracy; anything that cannot is worth at most ±0.005, i.e. inside leaderboard noise (SE 0.017).

Remaining untried, with honest expected value:
- **Ordinal / latent-evidence multi-task supervision** on the LightGBM-strength representation (not the weak MLP/TCN backbones) — the best of the rest, because it changes the *supervision*, not the architecture; expected +0.00x.
- **Pretrained time-series embeddings as extra features** — allowed by the rules; genuinely different representation; unknown, probably small on 2104 rows.
- **LightGBM HPO on the base+cross-site features** — never done; expected +0.002–0.005.
- **Site-pair gain-*magnitude* regression** as a correction to base gains (round 2 only tested the sign) — untested.

---

## 12. Data-level forensics: is the provided data what it claims, and is there a channel-level train/test shift?

Prompted by two candidate source datasets the user found. All checks run on the provided `train.csv`/`test.csv` only.

### 12.1 The 45 channels are genuine acc/gyro/mag
Mean lag-1 autocorrelation across the 8 frames, per channel position within a site (averaged over sites and rows):

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| declared | acc | acc | acc | gyro | gyro | gyro | mag | mag | mag |
| lag-1 autocorr | 0.854 | 0.945 | 0.915 | **−0.086** | 0.355 | **−0.139** | 0.919 | 0.954 | 0.952 |
| mean abs frame-to-frame step | 0.294 | 0.165 | 0.204 | 0.746 | 0.564 | 0.711 | 0.178 | 0.141 | 0.153 |

Gyroscope decorrelates between frames (rotation rate at 0.5 s spacing), magnetometer is the smoothest (slowly drifting field), accelerometer is in between. The declared layout is real; no channel block is synthetic or duplicated. Frame lags for acc are 0.854 / 0.803 / 0.801 (lag 1/2/3) — consistent with **point samples 0.5 s apart**, not averages of contiguous windows.

### 12.2 Values are continuous, not integer-derived
Per channel: ~10,000 distinct values, minimum gap ≈ 1e-5 on every channel (that is just the CSV's 5-decimal rounding), median gap ≈ 2e-4. Integer sensor counts would leave a coarse lattice (for ADC counts with σ≈50, a step near 0.02). There is none.

### 12.3 The five sites are distinct signals
Mean |correlation| between sites on the same channel: max off-diagonal **0.462** (RWrist↔RUpArm, same arm), then LThigh↔RWrist 0.357 (arm swing with gait), LThigh↔LAnkle 0.256; Waist is the most independent (0.07–0.15). Copies or linear mixes of fewer sensors would be near 1.0.

### 12.4 Both candidate datasets are refuted at data level
- **MotionData (`rightbody0417_test.csv`)**: 17 columns = Label, Outcome, 5 sensors × XYZ. Accelerometer only, raw integer counts (~1650–2200), 95-row windows, 10 gesture classes (run, punch, hug, jump, jumping_jack, hook, pick, chair, walking, turn). Cannot produce 9 channels per site.
- **WISDM**: 2 locations (phone in pocket, watch on wrist), accelerometer + gyroscope only, 20 Hz, 51 subjects, 18 activities recorded in separate 3-minute blocks. No magnetometer, no upper arm/waist/thigh/ankle, and no activity transitions (the target needs 13 samples before *and after* a boundary).

Fingerprint of the true source: 5 IMUs (acc+gyro+mag) at right wrist, right upper arm, waist, left thigh, left ankle; continuous recording with annotated transitions; ~76 participants, ~9 boundaries each.

### 12.5 NEW FINDING — RUpArm is out of range in test
Per-participant IQR across all 59 train groups vs the test set, on the same channel:

| Channel | train IQR min–max (59 groups) | pooled train | **test** | train groups at least that wide |
|---|---|---|---|---|
| **RUpArm acc-y** | 0.086 – 1.006 | 0.448 | **3.398** | **0 / 59** |
| **RUpArm mag-y** | 0.177 – 1.116 | 0.645 | **2.457** | **0 / 59** |
| Waist acc-x | 0.035 – 2.598 | 1.075 | 1.055 | 4 / 59 |
| RWrist acc-x | 0.483 – 1.843 | 1.123 | 1.181 | 22 / 59 |
| LAnkle gyro-x | 0.028 – 0.676 | 0.146 | 0.171 | 23 / 59 |

RUpArm acc-y in test is **10× the train median and 3.4× wider than the widest single train participant**; RUpArm mag-y is 2.2× wider than the widest. IQR is outlier-robust, so this is the *body* of the distribution, not a tail. Every other channel checked sits inside the normal between-participant range.

Separately, the heavy tail differs: test frames with |x|>8 are **310 of 327 at Waist**, almost all accelerometer; train's extremes are mostly gyroscope spread over Waist/LThigh/LAnkle. Excluding the 10.6% of test rows containing any |x|>8, the median test/train std ratio falls from 1.027 to **1.008** — so the headline "test std 1.19" is mostly heavy tails plus these two RUpArm channels, not a global rescaling.

**Why section 2's adversarial validation missed this**: that check compared *aggregate feature-group* AUC against a 17-participant pseudo-baseline, which absorbs per-channel range differences. A per-channel distributional test against the between-participant envelope is the sharper instrument and should be run on every challenge.

**Consequences.** RUpArm is the site with the worst recall (0.210, section 10.4), and its test distribution lies outside the training envelope, so any model must extrapolate there. This is a plausible contributor to the OOF 0.769 → LB 0.754–0.757 gap. Note it cannot be "fixed" with test statistics (banned); the compliant response is train-side robustness: clipping inputs at train-derived quantiles, or scale augmentation, both computed inside the training fold and applied per row. A stress test (simulating the observed distortion on held-out folds, with and without train-quantile clipping) was running when this section was written.

### 12.6 Stress test of the RUpArm shift — no fix worth shipping
Simulated the observed distortion on held-out fold rows only (RUpArm acc-y ×7.6, mag-y ×3.8, matching the measured IQR ratios; "mild" = ×3.0/×2.0), and compared the plain LightGBM against one whose inputs are clipped to the training fold's 0.5/99.5 percentiles (fitted on train rows only, applied per row — compliant).

| model | clean | test-like distortion | mild distortion |
|---|---|---|---|
| plain (base+cross-site) | **0.7640** (top-1 0.5646) | 0.7584 (0.5375) | 0.7611 (0.5537) |
| train-quantile clipping | 0.7625 (0.5570) | 0.7596 (0.5461) | 0.7615 (0.5537) |

- **The distortion costs only −0.0056 OOF** at the full measured magnitude. So the out-of-range RUpArm channels, though real and striking, are worth well under half of the 0.769 → 0.755 OOF-to-leaderboard gap; most of that gap remains ordinary leaderboard noise (SE 0.017).
- **Clipping is a wash**: −0.0015 on clean data, +0.0012 under distortion. Not shipped.

Interpretation: the models rely on RUpArm only weakly (its recall is 0.210 and, per section 10.4, substituting RWrist is a cheap error), so corrupting those two channels does comparatively little damage. A distribution shift matters only in proportion to how much the model leans on the shifted feature — worth checking that product, not just the shift magnitude, before engineering a defence.

---

## 13. Round 5: recommender-style ranking ideas — metric-aligned LambdaMART and per-site calibration

Two ideas that had not been tried, both motivated by the task being a ranking/recommendation problem rather than five independent regressions.

### 13.1 Per-site calibration — refuted
Sorting compares five scores against each other, so systematic per-site bias reorders rankings. Fitted a monotone (isotonic) map per site, predicted gain → true gain, on each fold's training rows and applied to the held-out fold.

| Member | raw | per-site isotonic |
|---|---|---|
| LightGBM (base+cross-site) | **0.7640** (top-1 0.5646) | 0.7628 (0.5627) |
| Transformer | **0.7448** | 0.7420 |
| v3 blend with both calibrated | **0.7689** | 0.7694 |

Refuted. The members' cross-site comparability is already adequate; monotone recalibration only discards information. (The uneven per-site recall of section 10.4 is the *correct* response to asymmetric gains, not a calibration fault.)

### 13.2 Metric-aligned LambdaMART — the closest call of the whole project, still refuted
LightGBM `LGBMRanker`, `objective="lambdarank"`, with **`label_gain` set to this competition's own gains** (0.10, 0.25, 0.45, 0.70, 1.00), `eval_at=[5]`, `lambdarank_truncation_level=5`, on long format (one row per window×site: the site's 54 stats, the same relative to the 5-site mean, the 5-site mean stats, site one-hot, lead; 168 features, 10520 rows). This differs from the round-1 LambdaRank test (0.7569), which used LightGBM's default exponential gains — a different objective from the scored metric.

Alone: **0.7610** (single seed), 0.7573 / 0.7583 on seeds 1 / 2, **0.7627** seed-averaged — below the LightGBM member's 0.7640 but a genuinely different model.

Blended into v3 (single seed), weight chosen in-sample:

| w_rank | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|
| OOF | 0.7715 | 0.7716 | 0.7718 | **0.7724** | 0.7721 | 0.7705 | 0.7682 |

That looked like the first real gain in five rounds: +0.0035 over v3, the entire weight curve above v3, and unlike the TCN the nested search gave the ranker **non-zero weight in all five folds** (0.4, 0.4, 0.4, 0.1, 0.5). But:

| Check | Result |
|---|---|
| Nested weight selection (single seed) | **0.7691** vs v3 0.7689 |
| Nested weight selection (3-seed average) | **0.7673** — *worse* than v3 |
| Per-fold delta at w=0.4 (seed-avg) | +0.0097, +0.0047, +0.0023, +0.0137, **−0.0219** |

**Diagnosis — why "4 of 5 folds improved" was not evidence.** Per-fold scores on fold 5: ranker 0.7409 / 0.7378 / 0.7483 across seeds, LightGBM member 0.7712, v3 0.7840 (v3's *best* fold). The ranker is systematically ~0.03 weaker on that participant group, so its contribution damages precisely where the blend is strongest, and the damage grows monotonically with its weight (−0.0007 → −0.0225 as w goes 0.1 → 0.6). Seed-averaging *raised* the ranker's standalone score (0.7610 → 0.7627) while leaving the fold-5 deficit intact, which distinguishes a **group-level weakness** from seed variance. Refuted; not shipped.

**Method note worth keeping:** when a candidate member improves most folds but loses one badly, re-run it with several seeds before believing either the gain or the loss. If seed-averaging improves the member alone but does not shrink the bad fold, the disagreement is structural (a participant group the member handles poorly), and nested weight selection will — correctly — refuse to pay for it.

### 13.3 Status
Five rounds, ~40 distinct configurations. **v3 (OOF 0.7689, LB 0.7543) stands as the final solution**; v2 (LB 0.7569) and v1 (LB 0.7563) are statistically indistinguishable from it. Nothing tested reaches the top-1 accuracy (~0.661) that 0.80 would require, and the best honest nested estimate anywhere in the project is 0.7691.

---

## 14. Round 6: sensor-orientation augmentation and metric-shaped loss weighting — refuted

Two further ideas, both with a concrete mechanism rather than hope.

**Setup.** LightGBM per-site on base+cross-site features. *Rotation augmentation*: for each training row and each site, sample one random 3-D rotation (axis uniform, angle ±25°) and apply it to that site's acc, gyro and mag triplets consistently (same rotation per site, as physics requires), then train on the original plus 2 rotated copies. *Gain weighting*: sample weight 1 + 2·[site's true gain ≥ 0.70], so errors on the top two positions count more, aligning the loss with the metric's steep position discounts. Each config scored on clean held-out rows **and** under the section-12.5 RUpArm distortion (acc-y ×7.6, mag-y ×3.8).

| Config | OOF | top-1 | under RUpArm stress |
|---|---|---|---|
| baseline | **0.7640** | **0.5646** | **0.7584** |
| rotation ×3 | 0.7596 | 0.5490 | 0.7536 |
| gain-weighted loss | 0.7622 | 0.5561 | 0.7527 |
| rotation ×3 + gain-weighted | 0.7625 | 0.5566 | 0.7490 |

Swapping each into v3's LightGBM slot: 0.7702 / 0.7698 / 0.7696 vs v3's 0.7689 — but per-fold deltas for the best of them are −0.0049, −0.0043, +0.0057, +0.0063, +0.0040 (mixed signs, 3 of 5 positive, weights not nested). By the round-5 fold-consistency rule this is noise, not a gain. Nothing shipped.

**Why rotation augmentation failed — a transferable lesson.** It was expected to buy robustness to sensor mounting, and it made the stress score *worse* (0.7536 vs 0.7584), failing on its own objective. Cause: the released data is **standardized per channel**, so each of the three axes has already been independently rescaled. A rotation applied in that space is not a physical rotation — it mixes axes whose relative scales have been destroyed, and with them the per-axis gravity direction that encodes posture (section 10.6 showed absolute levels are the most valuable signal, since removing them loses accuracy). **Geometric augmentation is unavailable on per-channel-standardized data unless the original relative axis scales can be restored.** The same caveat applies to any rotation-invariant architecture on this input.

**Why gain weighting failed.** Up-weighting the top positions costs calibration of the lower ones, and since the decode sorts by *expected gain across all five sites*, a distorted low end reorders the top just as effectively. Under this metric, uniform regression of all five gains is already the aligned objective.

### 14.1 Final status
Six rounds, ~45 configurations, one clean feature win (cross-site, +0.004 on the LightGBM member) and no further improvement. **v3 remains final: OOF 0.7689, LB 0.7543.** Best honest nested estimate anywhere in the project: 0.7691.

Still untested at the end: CatBoost and ExtraTrees (both lost to host memory exhaustion, never run), kernel ridge / GP-style small-sample models (effective n is 526 boundaries, not 2104 rows), hyperparameter search of any kind, feature selection among the 681 features, separate acc/gyro/mag encoders with learned fusion, ordinal P(rank ≤ k) supervision, transition-archetype mixtures, and pretrained time-series embeddings. Expected value of the lot, given six rounds of evidence: roughly +0.003 to +0.008, i.e. 0.772–0.777 — short of 0.80, which needs top-1 accuracy ≈ 0.661 against the current 0.5756.

---

## 15. Round 7: the leftovers — and the first validated improvement (v4)

Ran the model families that had never actually executed (two were lost to host memory exhaustion earlier, not refuted).

| Model (base+cross-site features, 5-fold) | OOF | top-1 | as v3's LightGBM member |
|---|---|---|---|
| **ExtraTrees** (300 trees, min_samples_leaf 2, max_features 0.2) | **0.7677** | **0.5651** | 0.7699 |
| CatBoost (MultiRMSE, 600 iters, depth 6) | 0.7613 | 0.5494 | 0.7675 |
| Kernel ridge (PCA-64, RBF, α=1) | 0.7547 | 0.5333 | 0.7657 |
| Kernel ridge (PCA-64, RBF, α=10) | 0.7412 | 0.5152 | 0.7599 |
| (reference) LightGBM | 0.7640 | 0.5646 | — |

**ExtraTrees is the strongest single model in the project** (0.7677 > LightGBM's 0.7640). Swapping it *in place of* LightGBM gives only 0.7699 with mixed per-fold signs — but adding it as a **fifth member** is different:

| w_ExtraTrees added to v3 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|
| OOF | 0.7707 | **0.7727** | 0.7727 | 0.7725 |
| top-1 | 0.5779 | **0.5832** | 0.5827 | 0.5794 |
| folds improved | 4/5 | 4/5 | 4/5 | 4/5 |

**Nested weight search over all five members** (weights chosen on 4 folds, scored on the 5th): **0.7727**, top-1 **0.5832**, versus v3's 0.7689 / 0.5756. Chosen weights per fold: (0.3, 0.2, 0.1, 0.4, 0.0) four times and (0.4, 0.4, 0.1, 0.4, 0.0) once — ExtraTrees takes 0.4 in **every** fold, CatBoost 0.0 in every fold.

### 15.1 Why this passed where five earlier candidates failed
The checklist built up over rounds 2–6, applied to every candidate:

| Test | TCN | LambdaMART | rotation aug | **ExtraTrees (v4)** |
|---|---|---|---|---|
| Beats v3 in-sample | +0.0009 | +0.0035 | +0.0013 | +0.0038 |
| Survives nested weight selection | ✗ 0.7689 | ✗ 0.7673 | not run | **✓ 0.7727** |
| Weight stable across folds | zeroed 4/5 | 0.1–0.5 | — | **✓ 0.4 in 5/5** |
| Worst fold | — | **−0.0219** | −0.0049 | **−0.0010** |
| Raises top-1 | 0.5813 | 0.5708 | 0.5490 | **✓ 0.5832** |

The decisive difference is that ExtraTrees' contribution is *uniform*: no participant group where it collapses, so nested selection pays for it in every fold.

### 15.2 v4 as shipped
`BLEND = {nn: 0.3, lgb: 0.2, knn: 0.1, et: 0.4}`, ExtraTrees added as `fit_predict_et` on the same base+cross-site features (300 trees, min_samples_leaf 2, max_features 0.2, fixed `n_jobs`, `random_state=0`).
- Full run 2m46s, CSV validated (592 rows, ids match, no duplicates/blanks), compliance grep clean, one comment.
- 208 of 592 rankings changed vs v3; top site changed in 34 rows.
- **Nested OOF 0.7727 vs v3's 0.7689 (+0.0038).** Against leaderboard noise of ±0.017 this may not show publicly, but it is the only change in seven rounds that survived every check.

**Method note:** a model that is *worse than a swap candidate* can still be the best *addition*. ExtraTrees replacing LightGBM scored 0.7699 with mixed signs; ExtraTrees alongside it scored 0.7727 with consistent signs. Always test a new member both ways before discarding it.

---

## 16. Round 8: hyperparameter search, feature selection, and the limits of large member pools — all negative

### 16.1 Nested HPO and feature pruning (the last untested items)
Six LightGBM configurations chosen on an inner 2-fold split of each training fold, then refit and scored on the held-out fold; plus importance-based pruning to the top 150 / 300 of 681 features.

| Variant | OOF | top-1 |
|---|---|---|
| untuned LightGBM (shipped) | **0.7640** | **0.5646** |
| nested HPO | 0.7620 | 0.5518 |
| top-150 features | 0.7618 | 0.5623 |
| top-300 features | 0.7599 | 0.5585 |

All worse than the untuned default. The HPO also chose a *different* configuration in each fold (`num_leaves` 7/15/31, `colsample` 0.3/0.5), which is itself evidence that the differences between configurations are noise rather than signal. Tuning and feature selection are now tested and closed.

### 16.2 Adding them to v4 — nothing survives
Each candidate added to the shipped v4 blend (in-sample weight, per-fold signs):

| Added at w=0.1 / 0.2 | folds improved |
|---|---|
| nested-HPO LightGBM | 1/5, 0/5 |
| top-150 features | 3/5, 2/5 (deltas in the 4th decimal) |
| top-300 features | 1/5, 1/5 |
| CatBoost | 1/5, 0/5 |
| metric-aligned ranker | 2/5, 3/5 (worst fold −0.0116) |

### 16.3 A large member pool overfits the weight search
Nested greedy over all **9** members (weights built on 4 folds, scored on the 5th):

- **0.7673**, top-1 0.5637 — *worse* than v4's nested 0.7727 / fixed-weight 0.7738. Per-fold delta vs v4: −0.0011, −0.0006, −0.0111, +0.0038, −0.0252 (1/5 up).
- The chosen weights are unstable across folds (the ranker gets 0.1 in one fold and 0.6 in another), the signature of selection overfitting with correlated members and only 59 participant groups.
- **ExtraTrees appears in all five folds' picks (0.2–0.4)** — independent confirmation that it, and not the pool, is the real contributor.

**Lesson:** with ~59 groups, a per-fold weight search over many correlated members fits the selection itself. A *small, fixed, stability-checked* weight set (v4: four members, weights identical in 4/5 nested folds) beats an adaptive search over a big pool by 0.005+.

### 16.4 Final state
**v4 is the final solution**: `BLEND = {nn 0.3, lgb 0.2, knn 0.1, et 0.4}`, nested OOF **0.7727** (fixed-weight 0.7738), top-1 **0.5841**, vs v3's 0.7689 / 0.5756. Eight rounds, ~55 configurations; the only two validated wins in the project are **cross-site features** (+0.004 on the LightGBM member) and **ExtraTrees as a fourth member** (+0.0038 nested). Everything else — temporal models, ranking losses, calibration, specialists, augmentation, invariance, tuning, pruning, larger ensembles — is refuted with numbers above.

---

## 17. Round 9: why v4 worked on the leaderboard — and the limit of that lever

**v4 leaderboard: 0.7680 (script run) / 0.7640 (CSV upload)** vs v3 0.7543, v2 0.7569, v1 0.7563. Those two v4 numbers are *the same solution scored twice* (the grader re-ran the script; the torch member landed differently on its hardware), which gives a direct measure of **run-to-run variation: 0.0040**. Treat any candidate gain below that as unmeasurable.

The leaderboard gain (+0.0137 over v3) outran the validation gain (+0.0038). Part is luck — it is ~0.8 of one standard error — but the mechanism is real and measurable.

### 17.1 The mechanism: ExtraTrees raises the *floor*, not the mean

| Member | OOF | per-group std | **worst group** | groups won (of 59) |
|---|---|---|---|---|
| **ExtraTrees** | 0.7677 | **0.0691** | **0.6200** | **20** |
| LightGBM | 0.7640 | 0.0704 | 0.5759 | 18 |
| kNN | 0.7480 | 0.0806 | 0.4913 | 7 |
| Transformer | 0.7448 | 0.0721 | 0.5762 | 14 |
| v4 blend | **0.7738** | **0.0668** | 0.6015 | — |

Split the 59 participant groups into quartiles by LightGBM's score: ExtraTrees beats LightGBM by **+0.0118 on the hardest quartile** and *loses* by 0.0075 on the easiest. It is not more accurate on average — it fails less badly on difficult participants. With only 17 test participants, that floor is worth more than mean accuracy. Randomized split thresholds depend far less on participant-specific feature values than boosted splits do, which is precisely what an unseen-participant test punishes.

### 17.2 The lever is exhausted — four ways of pushing it all fail
1. **More ExtraTrees weight**: 0.4 is optimal. 0.5 → 0.7723, 0.6 → 0.7712, 0.7 → 0.7701, alone → 0.7677. Dropping any other member also loses (no-nn 0.7683, no-lgb 0.7702, no-knn 0.7728).
2. **Stronger randomized trees**: ET 600 trees/leaf 1 → 0.7683 alone, 0.7742 in the blend (+0.0004, noise). ET at leaf 5, max_features 0.1 or 0.4 → all ≤ baseline. LightGBM's own `extra_trees=True` mode → 0.7594, clearly worse. Two randomized members together → ≤ v4.
3. **RandomForest**: 0.7676 alone, **0.7757 in the ExtraTrees slot** — the batch's best number, but only 3/5 folds up with two negatives, and RF's per-row scores correlate **0.897** with ExtraTrees, i.e. a near-substitute rather than a new source of diversity. Nested over five members: 0.7735 (RF takes 0.4 in 5/5 folds, ExtraTrees is zeroed in 3/5 — they trade places), vs v4's fixed 0.7738. No gain.
4. **Floor-aware weight selection** (maximise the 20th-percentile participant instead of the mean): raises the worst group to 0.6229 but drops the mean to 0.7698 and top-1 to 0.5684. Since the metric *is* the mean over participants, trading mean for floor is the wrong trade.

### 17.3 What the 17-participant draw implies
2000 bootstraps of 17 groups from the v4 OOF: expected **0.7735, 5–95% range 0.7469–0.8011**. Both v4 leaderboard scores (0.7680, 0.7640) sit inside that band, slightly below the centre. Note 0.80 lies *within* the luck band for the model as it already stands — the participant draw moves the score by more than any modelling change made in nine rounds.

### 17.4 Final
**v4 is final**: `BLEND = {nn 0.3, lgb 0.2, knn 0.1, et 0.4}`, OOF 0.7738 (nested 0.7727), top-1 0.5841, LB 0.7640–0.7680.

**Transferable lesson for participant-held-out (or any group-held-out) tasks:** rank candidate members by their *worst-group* score, not only their mean. Randomized-split ensembles (ExtraTrees, RandomForest) can beat boosted trees on transfer while being no better — or slightly worse — on average, and that advantage shows up on the leaderboard rather than in a mean-OOF comparison. But once one such member is in the blend, adding a second correlated one (ET+RF, r = 0.897) buys nothing.
