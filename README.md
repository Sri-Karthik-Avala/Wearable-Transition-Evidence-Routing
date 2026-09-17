# Wearable Transition Evidence Routing

| | |
| --- | --- |
| Final rank | #22 |
| Domain | Recommendation |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Creator | yenwee0804 |
| Solutions submitted | 4 |
| Last submission | 2026-09-13 |

## Problem statement

### Overview

Adaptive wearables sometimes need to reduce radio, battery, or inference cost by deciding which physical sensor placements deserve priority. Each row gives an ordered four-second context from five real multi-site IMU placements, ending before an annotated activity boundary. Your task is to produce a complete priority ranking of the five placements whose signals are most useful for detecting the impending change: the first site is the primary route, followed by a ranked fallback queue.

This is a recommendation and ranking task where the input contains no post-boundary measurements and no target evidence; the hidden target is derived from the real signal change at that boundary. The test participants are disjoint from the labeled participants, so a successful model must transfer placement evidence across people and lead offsets rather than memorize participant or row order.

The input is an ordered sensor sequence, so a temporal encoder or sequence-to-sequence model is a valid modeling choice. The scored output is nevertheless a site-priority ranking, not a generated activity string. The four lead offsets expose how early a placement decision remains useful, while the grouped validation file supports participant-disjoint model selection. This benchmark is calibrated for lightweight or medium learned models on a subject-held-out dataset; it is not a from-scratch pretraining task.

### Task boundary and practical proxy

Published neighboring work such as P2LHAP uses wearable patches to output activity labels, including future activity labels. This benchmark does not score activity identities, transition identities, or a generated label sequence. It scores whether a model can rank all physical placements by the clearest normalized signal change soon after a boundary, which isolates adaptive sensing and fallback routing from activity recognition itself. The target is a benchmark proxy for a controller that maintains a full sensing-priority queue and can keep the highest-ranked placements at higher rate; it is not presented as a clinical policy or a complete downstream transition-detection system.

### Dataset

### File descriptions

- `train.csv` -- 2,104 labeled windows. Each row contains an opaque id, the known lead offset, the serialized sensor sequence, and the complete ordered site-priority target in the canonical `prediction` column.
- `test.csv` -- 592 unlabeled windows with the same input columns as `train.csv` except for the train-only `prediction` target.
- `sample_submission.csv` -- A valid 592-row example submission with a randomized permutation of the five body-site labels for every test id.
- `validation_groups.csv` -- A 2,104-row mapping from each labeled id to an opaque participant-group id. Keep all windows from one group in the same validation fold.

There are 526 independent training boundaries and 148 independent hidden-test boundaries; each boundary contributes four correlated lead-offset rows. The raw release also contains `fold_assignments.csv`, but that source metadata is retained for provenance only and does not define this benchmark split. The materializer uses fixed subject sets for 8 public validation participants and 17 hidden-test participants, while `validation_groups.csv` exposes the resulting participant groups for train-only validation.

### Column descriptions

- `id` (string) -- An opaque, unique identifier for one window. It has no semantic ordering.
- `lead_offset_s` (float) -- Known input feature describing how far before the boundary the four-second context ends. It is one of `1.0`, `3.0`, `5.0`, or `7.0` seconds and is available in both train and test.
- `sensor_sequence` (string) -- A compact JSON list of 360 finite numbers. Reshape it row-major to `(8, 45)` to recover eight ordered frames and 45 channels.
- `prediction` (string, train only) -- All five distinct body-site labels separated by `>`, ordered from primary route through the fallback queue. Valid labels are `RWrist`, `RUpArm`, `Waist`, `LThigh`, and `LAnkle`.
- `validation_group` (string, in `validation_groups.csv`) -- An opaque group id shared by all labeled windows from one participant. It is for grouped validation only and is not a predictive feature.

The sequence channel order is fixed:

- placement 1: accelerometer `x,y,z`, gyroscope `x,y,z`, magnetometer `x,y,z`
- placement 2: accelerometer `x,y,z`, gyroscope `x,y,z`, magnetometer `x,y,z`
- placement 3: accelerometer `x,y,z`, gyroscope `x,y,z`, magnetometer `x,y,z`
- placement 4: accelerometer `x,y,z`, gyroscope `x,y,z`, magnetometer `x,y,z`
- placement 5: accelerometer `x,y,z`, gyroscope `x,y,z`, magnetometer `x,y,z`

The sequence values are standardized channel-wise using labeled training rows only. The repeated lead offsets are separate rows with equal evaluation weight; do not infer a participant or boundary order from the opaque id.

### Evaluation

Submissions are scored with worst-to-ideal normalized graded NDCG@5, a maximize metric in `[0, 1]`. The target ranks have fixed relevance gains `(1.0, 0.70, 0.45, 0.25, 0.10)`, reflecting a primary route followed by progressively less preferred fallbacks. All five submitted positions therefore affect the score, while an error near the top is more costly than an error near the bottom. The usual NDCG value is rescaled between the mathematically worst reverse ranking and the ideal ranking; this removes the high random-permutation floor caused by every valid site having a positive fallback value.

For each row, the grader parses a permutation of all five valid site labels. It assigns each submitted site the gain of its position in the hidden target, computes DCG for the submitted order, and rescales it between the worst reverse-order DCG and the ideal DCG. The final score is the unweighted mean across hidden rows, including all four lead offsets.

The target used to construct the private answers is a real-signal evidence ranking. For each annotated boundary, the organizer compares 13 real samples immediately before the boundary with the first 13 real samples after it, stopping at the next observed boundary. For each body site, the evidence value is the mean over its nine channels of `abs(post_mean - pre_mean) / (pre_std + 0.05)`. The small scale floor prevents an almost-constant channel from dominating through division by a near-zero pre-boundary standard deviation. Sites are sorted from highest to lowest evidence with the fixed site order as the deterministic tie-break, and the complete five-site order forms the train-only `prediction` value. The post-boundary samples used for this target are never included in public `test.csv`.

```
import math

import numpy as np

SITE_LABELS = ("RWrist", "RUpArm", "Waist", "LThigh", "LAnkle")

RANK_GAINS = (1.0, 0.70, 0.45, 0.25, 0.10)

IDEAL_DCG = sum(gain / math.log2(rank + 2.0) for rank, gain in enumerate(RANK_GAINS))

WORST_DCG = sum(gain / math.log2(len(RANK_GAINS) - rank + 1.0) for rank, gain in enumerate(RANK_GAINS))

def parse_ranking(value):

    sites = tuple(str(value).split(">"))

    if len(sites) != len(SITE_LABELS) or len(set(sites)) != len(SITE_LABELS) or set(sites) != set(SITE_LABELS):

        raise ValueError("a ranking must contain all five distinct valid sites")

    return sites

def ndcg_at_5(true_ranking, predicted_ranking):

    true_sites = parse_ranking(true_ranking)

    predicted_sites = parse_ranking(predicted_ranking)

    gains = {site: gain for site, gain in zip(true_sites, RANK_GAINS)}

    dcg = sum(

        gains[site] / math.log2(rank + 2.0)

        for rank, site in enumerate(predicted_sites)

    )

    return (dcg - WORST_DCG) / (IDEAL_DCG - WORST_DCG)

score = float(np.mean([

    ndcg_at_5(true_ranking, predicted_ranking)

    for true_ranking, predicted_ranking in zip(true_rankings, predicted_rankings)

]))
```

Here `true_rankings` are the hidden `prediction` values and `predicted_rankings` are the submitted values aligned by `id`. `RANK_GAINS`, `IDEAL_DCG`, and `WORST_DCG` are fixed constants; a perfect ranking scores `1.0` and the reverse ranking scores `0.0`. The decreasing gains and DCG discounts make the rescaled score sensitive to all five positions. A valid score is always between `0` and `1`; malformed rankings, missing ids, duplicate ids, and unknown site labels are rejected rather than silently remapped.

### Submission

Submit a UTF-8 CSV with exactly one prediction for every row in `test.csv`.

- `id` (string) -- Copy each opaque id from `test.csv` exactly once.
- `prediction` (string) -- Your complete ordered ranking in the exact form `site1>site2>site3>site4>site5`, using every label from `RWrist`, `RUpArm`, `Waist`, `LThigh`, and `LAnkle` exactly once.

Example:

```
id,prediction

wte_56484d18a23e2e823eac652bf1c2ce9b,RWrist>Waist>LAnkle>RUpArm>LThigh

wte_cf21f7a287554f36bb986ba8763a63ed,LThigh>RUpArm>RWrist>Waist>LAnkle
```

### Requirements

- The file must contain exactly 592 data rows, matching `test.csv`.
- Every test id must appear exactly once; do not add or remove ids.
- The column names must be exactly `id,prediction`.
- Every prediction must be a non-empty, non-NaN string containing all five distinct valid site labels separated by exactly four `>` characters.
- Use the matching `sensor_sequence` and `lead_offset_s` row for each id.

### What Not To Use

- Do not reverse-map opaque ids to participant names, capture order, or boundary order. Those identifiers are deliberately non-semantic.
- Do not use `validation_group` as a model feature or infer the source participant from its opaque value.
- Do not use post-boundary samples, future frames, or any signal not present in the row's `sensor_sequence`; those samples define the hidden target but are not solver inputs.
