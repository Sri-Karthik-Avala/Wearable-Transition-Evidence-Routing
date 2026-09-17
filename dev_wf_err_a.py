"""KEY=err  error analysis of per-site LightGBM baseline."""
import time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import wf_harness as h
warnings.filterwarnings("ignore")
t0 = time.time()
d = h.load(); X, R, lead, groups = d["X"], d["R"], d["lead"], d["groups"]
n = len(X); F = h.folds(groups); SITES = h.SITES; LEADS = h.LEADS
TYPES = ("acc", "gyro", "mag")


def site_stats(X):
    n = len(X)
    Xs = X.reshape(n, 8, 5, 9).transpose(0, 2, 1, 3)
    norms = np.stack([np.linalg.norm(Xs[..., k:k + 3], axis=-1) for k in (0, 3, 6)], -1)
    parts = [Xs.std(2), Xs.mean(2), Xs[:, :, -1] - Xs[:, :, 0], np.abs(np.diff(Xs, axis=2)).mean(2),
             Xs[:, :, -2:].mean(2) - Xs[:, :, :-2].mean(2), norms.std(2), norms.mean(2),
             np.abs(np.diff(norms, axis=2)).mean(2)]
    return np.concatenate(parts, -1).astype(np.float32)


def flat(X, l):
    Fs = site_stats(X); n = len(Fs)
    rel = Fs / (np.abs(Fs).mean(1, keepdims=True) + 1e-6)
    return np.concatenate([Fs.reshape(n, -1), rel.reshape(n, -1), X[:, -1], l[:, None]], 1)


A = flat(X, lead)
S = np.zeros((n, 5), np.float32)
for tr, va in F:
    for s in range(5):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, random_state=s, n_jobs=1, deterministic=True,
                              force_row_wise=True, verbose=-1)
        m.fit(A[tr], R[tr, s]); S[va, s] = m.predict(A[va])
    print("  fold done t", round(time.time() - t0, 1), flush=True)
h.report("err__lgb_base", S, R, lead)
print("boundary_se", round(h.boundary_se(S, R, groups), 4), "t", round(time.time() - t0, 1))
sc = h.row_scores(S, R)

# ---------- top-1 confusion ----------
ttop = R.argmax(1); ptop = S.argmax(1)
print("\n== top-1 accuracy", round((ttop == ptop).mean(), 4))
cm = pd.crosstab(pd.Series([SITES[i] for i in ttop], name="true"), pd.Series([SITES[i] for i in ptop], name="pred"))
print(cm)
print("\n== score / top1 acc by TRUE top site")
for s in range(5):
    mk = ttop == s
    print(f"  {SITES[s]:7s} n={mk.sum():4d} score={sc[mk].mean():.4f} top1acc={(ptop[mk]==s).mean():.3f} predfreq={(ptop==s).mean():.3f} truefreq={mk.mean():.3f}")
print("\n== by lead: score, top1acc")
for l in LEADS:
    mk = lead == l
    print(f"  lead {l}: score={sc[mk].mean():.4f} top1={(ttop[mk]==ptop[mk]).mean():.3f}")
g = pd.Series(sc).groupby(groups).mean()
print("\n== per-group score: mean %.4f std %.4f min %.4f max %.4f q10 %.4f q90 %.4f" % (g.mean(), g.std(), g.min(), g.max(), g.quantile(.1), g.quantile(.9)))
print("   group size-score corr", round(np.corrcoef(pd.Series(sc).groupby(groups).size().values, g.values)[0, 1], 3))

# ---------- loss decomposition ----------
loss = 1 - sc
# oracle-fix top1: put true top first, keep remaining predicted order
S2 = S.copy(); S2[np.arange(n), ttop] = 1e9
loss_fix1 = 1 - h.row_scores(S2, R)
# oracle top1 + top2
S3 = S2.copy(); sec = np.argsort(-R, 1)[:, 1]; S3[np.arange(n), sec] = 1e8
loss_fix2 = 1 - h.row_scores(S3, R)
print("\n== loss decomposition: total loss %.4f | after oracle top1 %.4f (top1 share %.1f%%) | after oracle top1+2 %.4f" % (
    loss.mean(), loss_fix1.mean(), 100 * (loss.mean() - loss_fix1.mean()) / loss.mean(), loss_fix2.mean()))
print("   score if top1 correct (subset) %.4f, if wrong %.4f" % (sc[ttop == ptop].mean(), sc[ttop != ptop].mean()))
# rank of true top in predicted order
pr = np.argsort(np.argsort(-S, 1), 1)
print("   predicted position of true top site:", np.bincount(pr[np.arange(n), ttop], minlength=5) / n)
# pairwise accuracy by true rank pair
pair = np.zeros((5, 5)); cnt = np.zeros((5, 5))
Rr = np.argsort(np.argsort(-R, 1), 1)
for i in range(5):
    for j in range(5):
        if i == j: continue
        # rows where site i has true rank a and site j true rank b
        for a in range(5):
            for b in range(a + 1, 5):
                mk = (Rr[:, i] == a) & (Rr[:, j] == b)
                pair[a, b] += (S[mk, i] > S[mk, j]).sum(); cnt[a, b] += mk.sum()
print("   pairwise order accuracy by true-rank pair (a<b):")
for a in range(5):
    print("    ", " ".join(f"{pair[a,b]/cnt[a,b]:.2f}" if cnt[a, b] else "  - " for b in range(5)))

# ---------- single-statistic scorers ----------
Xs = X.reshape(n, 8, 5, 9)  # n,t,site,ch
def by_type(v):  # v: n,5,9 -> dict type-> n,5
    return {t: v[..., 3 * k:3 * k + 3].mean(-1) for k, t in enumerate(TYPES)}
stats = {}
std = Xs.std(1); stats.update({f"std_{k}": v for k, v in by_type(std).items()})
ld = np.abs(Xs[:, -1] - Xs[:, -2]); stats.update({f"lastdiff_{k}": v for k, v in by_type(ld).items()})
ad = np.abs(np.diff(Xs, axis=1)).mean(1); stats.update({f"absdiff_{k}": v for k, v in by_type(ad).items()})
ldv = np.abs(Xs[:, -1] - Xs[:, :-1].mean(1)); stats.update({f"lastdev_{k}": v for k, v in by_type(ldv).items()})
z = ldv / (Xs[:, :-1].std(1) + 0.05); stats.update({f"lastz_{k}": v for k, v in by_type(z).items()})
early = Xs[:, :4].std(1); late = Xs[:, 4:].std(1)
stats.update({f"lateovearly_{k}": v for k, v in by_type(np.log((late + 1e-3) / (early + 1e-3))).items()})
stats.update({f"negstd_{k}": -v for k, v in by_type(std).items()})
stats["std_all"] = std.mean(-1); stats["absdiff_all"] = ad.mean(-1); stats["lastz_all"] = z.mean(-1)
print("\n== single-stat scorer NDCG (rank sites by stat, higher first) per lead | pooled per-site spearman(stat, gain) mean over sites")
rows = []
for k, v in stats.items():
    scs = h.row_scores(v, R)
    sp = np.mean([spearmanr(v[:, s], R[:, s])[0] for s in range(5)])
    # within-site-centered: remove site mean effect -> how much does stat add beyond site identity
    rows.append((k, scs.mean(), *[scs[lead == l].mean() for l in LEADS], sp))
df = pd.DataFrame(rows, columns=["stat", "all", "l1", "l3", "l5", "l7", "sp_site"]).sort_values("all", ascending=False)
print(df.round(4).to_string(index=False))
# site-prior baseline
prior = np.tile(R.mean(0), (n, 1)); print("prior-order NDCG", round(h.row_scores(prior, R).mean(), 4))
# per-site spearman per type per lead for std (quietness)
print("\n== per-site spearman(std_type, gain) by lead  (negative = quiet sites get high gain)")
for k in TYPES:
    v = by_type(std)[k]
    line = []
    for l in LEADS:
        mk = lead == l
        line.append(np.mean([spearmanr(v[mk, s], R[mk, s])[0] for s in range(5)]))
    print(f"  {k}: " + " ".join(f"{x:+.3f}" for x in line))
print("\n== per-site spearman(lastz_type, gain) by lead")
for k in TYPES:
    v = by_type(z)[k]
    print(f"  {k}: " + " ".join(f"{np.mean([spearmanr(v[lead==l, s], R[lead==l, s])[0] for s in range(5)]):+.3f}" for l in LEADS))

# ---------- ceilings ----------
bid = pd.Series(groups).astype(str) + "|" + pd.Series(d["pred_str"])
bsize = bid.map(bid.value_counts()).values
m4 = bsize == 4
print("\n== boundary keys (group,target): rows with unique 4-lead key %.3f" % m4.mean())
# window overlap: for 4-row boundaries, compare lead1 vs lead3 windows at frame shifts
sub = pd.DataFrame(dict(b=bid[m4].values, l=lead[m4], i=np.where(m4)[0]))
piv = sub.pivot(index="b", columns="l", values="i")
for la, lb in [(1., 3.), (1., 5.), (3., 5.), (5., 7.)]:
    ia, ib = piv[la].values, piv[lb].values
    res = []
    for sh in range(0, 8):
        if sh == 0:
            res.append(np.abs(X[ia] - X[ib]).mean())
        else:
            res.append(np.abs(X[ib][:, sh:] - X[ia][:, :-sh]).mean())  # lead b earlier window: b frames shifted later?
    res2 = [np.abs(X[ia][:, sh:] - X[ib][:, :-sh]).mean() for sh in range(1, 8)]
    print(f"  lead{la} vs lead{lb}: mean|diff| shift ib[sh:]-ia[:-sh] " + " ".join(f"{x:.3f}" for x in res) + " | ia[sh:]-ib[:-sh] " + " ".join(f"{x:.3f}" for x in res2))
# disagreement among 4 lead rows (OOF)
bt = pd.DataFrame(dict(b=bid.values, p=ptop))[m4].groupby("b").p.nunique()
print("  fraction of boundaries where 4 lead rows disagree on top-1: %.3f ; mean #distinct tops %.2f" % ((bt > 1).mean(), bt.mean()))
# pooling across lead rows of same boundary (oracle diagnostic only)
Sz = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)
pool = pd.DataFrame(Sz).groupby(bid.values).transform("mean").values
print("  boundary-pooled (4-lead mean) OOF score %.4f vs row %.4f" % (h.row_scores(pool, R).mean(), sc.mean()))
w1 = np.where(lead == 1, 1.0, 0.0)
# score of best single lead rows: lead-1 prediction applied to all rows of boundary
l1 = pd.DataFrame(Sz * w1[:, None]).groupby(bid.values).transform("sum").values
print("  lead1-prediction broadcast to boundary: %.4f" % h.row_scores(l1, R)[lead != 1].mean(), "(on leads 3/5/7 rows)")
# kNN in standardized feature space (fold-wise), average neighbors' R
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
Fk = site_stats(X).reshape(n, -1)
for k in (5, 20, 50):
    Sk = np.zeros_like(S)
    for tr, va in F:
        sc_ = StandardScaler().fit(Fk[tr]); nn_ = NearestNeighbors(n_neighbors=k).fit(sc_.transform(Fk[tr]))
        _, idx = nn_.kneighbors(sc_.transform(Fk[va])); Sk[va] = R[tr][idx].mean(1)
    print(f"  kNN k={k} on site-stats: OOF {h.row_scores(Sk, R).mean():.4f}")
# target consistency: kNN on the TRUE target within train folds (how clustered targets are) -> prior spread
print("  target entropy: distinct orders %d of 120; top-5 orders cover %.3f" % (len(set(d['pred_str'])), pd.Series(d['pred_str']).value_counts().head(5).sum() / n))
print("done t", round(time.time() - t0, 1))
