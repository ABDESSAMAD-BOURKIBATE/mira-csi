# MIRA-CSI model card

**Developer and research lead:** Eng. Abdessamad Bourkibate<br>
**Task:** offline classification of recorded Wi-Fi CSI into seven activity labels<br>
**Status:** v0.3.0 experimental research software, unreviewed and without an assigned DOI. The current entry point is spectral E; the original temporal model and earlier evidence are retained below.

## Current variant: spectral E

[`spectral/`](spectral/) provides **MIRA-CSI spectral variant E (experimental)**,
developed by Abdessamad Bourkibate. It uses 360 observed-endpoint lagged-increment
features, a 360–64–32 GELU MLP, dropout and a seven-logit head: **25,415 trainable
parameters, with no temporal CNN**. It is distinct from the 40,871-parameter
temporal model described in the retained card below.

### Input, loading and output contract

The input is an unnormalized processed `(18, 500)` **log-mean-power18** tensor:
five-exported-subcarrier groups, three receive antennas and 500 normalized
valid-packet positions. It is not raw complex CSI, dBm, physical time or the
original amplitude representation. The optional `(500,)` mask is Boolean or
finite binary, with 1 indicating an observed position. Mask validation precedes
Boolean conversion. Non-finite observed values, invalid dimensions, numerical
feature failures and non-finite features/logits/probabilities are rejected.
Hidden numeric NaN/Inf fills do not enter the observed-only features; an empty
record is rejected because total loss was not evaluated.

The public wrapper verifies the pinned protocol/freeze, checkpoint and saved
validation/receipt hashes before restricted Torch loading. Class order and the
source-validation temperature must agree with the frozen records. Modules load
from the same verified package root. Thirteen interface tests include invalid
masks, hidden-fill feature independence, extreme finite-value overflow, mocked
non-finite outputs, verified-root loading and hash rejection. These tests check
software behaviour, not recognition accuracy. Hashes establish content
consistency, not authenticity or external validity. [Commands and exact files](spectral/README.md).

The first held E1 example is retained without selecting for success. Its true
label is `sit`; E predicts `pick_up` when complete and `stand` under its B50
mask. The two saved CLI replays match earlier batched probabilities within the
declared float32 tolerance. Later finite-value/root-loading hardening was
checked without additional real-record model inference.

### Current evidence and its limits

The extension uses 75 source fits: A–E × three outer folds × five seeds. The
additional MiniRocket power18 reference has 15 configurations, including three
reused source fits and 12 new ones. Both use the same power18 tensor, roles,
seeds and masks. E is loss-augmented; the reference is clean-trained with
source-band-mean fill and no explicit mask. This aligns input and prediction
roles, not the complete training/repair policy.

| Held condition | E accuracy (%) | E macro F1 (%) | Reference accuracy (%) | Reference macro F1 (%) |
|---|---:|---:|---:|---:|
| Complete | 75.175 ± 1.005 | 74.607 ± 1.207 | 53.841 ± 0.938 | 53.868 ± 1.009 |
| Random 50% | 73.822 ± 0.942 | 73.329 ± 1.105 | 22.305 ± 1.102 | 16.112 ± 0.841 |
| Contiguous 50% | 60.597 ± 1.375 | 59.159 ± 1.610 | 19.406 ± 2.330 | 15.636 ± 4.147 |

Masks are averaged within each training seed before computing mean and sample
SD across five seeds. These SDs are not population confidence intervals. NLL
and summed multiclass Brier, all negative controls and full fold/domain scores
are retained in the [32-page manuscript](paper/manuscript.pdf),
[four-page supplement](paper/supplementary-results.pdf) and v0.3.0 evidence
assets linked from the [README](README.md#evaluation). Under B50, E's NLL/Brier
are 1.160 ± 0.014 / 0.552 ± 0.011; the reference's are
4.852 ± 1.764 / 1.275 ± 0.168. E still has B50 pooled familiar accuracy
53.325 ± 1.480%, and two pooled held training seeds are below 60% accuracy.
Pooled familiar scores can repeat recordings and participants across outer
folds; those prediction occurrences are not independent new samples.

A/B hybrids remain weaker on complete held data than the reference:
48.302 ± 2.627% and 48.048 ± 2.559%. The original primary A−B B50 macro-F1
contrast remains +0.543 ± 0.654 percentage points, with four positive and one
negative training seed. The additional, post-hoc E−reference contrast is
+43.524 ± 3.639 points. Its [39.879, 47.054] participant-bootstrap range is
descriptive and conditional on fixed rooms, models, seeds and masks; it is not
corrected for post-hoc selection and does not cover new-room or new-training-data
uncertainty. E was a prespecified arm before its evaluation but was highlighted
after the results; historical target outcomes had already influenced development.

The construction is attributed to known covariance/temporal precedents in
SFA/SOBI and earlier CSI eigen-feature work in PADS. Both A and B have ideal
full-effective-support congruence invariance; matching supplies an endpoint
reference and [0,2] bound. Numerical truncation can break that invariance, even
at unchanged retained rank. A−B changes endpoint weighting and centering even
on complete data; E's feature-only comparator with B's denominator was not
tested. Consequently these results do not isolate a matching mechanism or a
causal architectural effect, establish SOTA, or support a patent/first-in-field
claim. Independent data and training-policy-matched controls remain necessary.

All source weights and calibration were frozen before held/familiar model
inference in this execution. Static per-record conversion and hash reads are
distinct from target fitting or model inference. Loss is imposed after
packet-index normalization, and different cohorts occupy different environments.
No experiment here validates continuous live sensing, body imaging, vital signs
or a causal room-only effect.

## Earlier temporal model: retained documentation

**The remainder of this card describes the original temporal predictor and
v0.1.0–v0.2.1 studies. Its amplitude input, 40,871-parameter architecture,
checkpoints and numerical results must not be relabelled as spectral E.**

## Input and output

The predictor consumes a processed `(18, 500)` amplitude array and optionally
an observation mask. It returns seven model probabilities and their highest
scoring activity label. The evaluation uses 1,260 public recordings from 30
participants in three environments. Each environment contains a different
participant cohort.

These inputs require the documented CSI acquisition and preprocessing system.
An ordinary laptop RSSI reading is insufficient. The evaluated classifier does
not produce a body image, pose mesh, thermal image or a live view of a user.
Water content, vital signs and clinical fall diagnosis are outside its task.

## Computation

For `m[t] = 1` at retained positions, define
`d[t] = min_{s:m[s]=1} |t-s| / 500`. Offline repair uses retained observations
or a source-training mean. Per-band means and standard deviations are fitted
on source-training recordings only. The input concatenates 18 standardized
amplitude streams, the observation mask and the nearest-support distance.

Three temporal convolutions use 32/64/128 channels, kernel lengths 7/5/3,
stride two, GroupNorm and GELU. Adaptive temporal binning of `m` supplies
coverage weights `w[j] = max(coverage[j], 0.05)`. The representation is
`sum_j w[j] h[j] / sum_j w[j]`; dropout and a linear head produce seven logits.
There are **40,871 nominal trainable parameters**. The amplitude-only CNN
has 40,423.

Coverage is a heuristic for observation availability. It is neither calibrated
prediction confidence nor measured RF reliability. Because GroupNorm aggregates
across time, a bin's coverage is not the observed fraction of its feature's
complete dependency set. Distance and interpolation may use later retained
positions; this is an offline procedure.

Training minimizes seven-class cross-entropy with AdamW. Source validation
selects the checkpoint and scalar temperature. Target labels are used for
scoring only. The [original protocol](protocol.json), its
[compute amendment](protocol_amendment_01.json), and the
[component-control protocol](studies/component_removal/protocol.json) distinguish
the evaluated stages.

## What the contribution tests

MIRA-CSI implements a particular combination of explicit missingness inputs
and coverage-weighted pooling. Masks, time gaps, CNNs and missing-CSI recovery
have prior art; see [PRIOR-ART.md](PRIOR-ART.md). Authorship of this implementation
does not establish invention of those ideas or ownership of upstream methods.

The initial comparison changes three additions together. A subsequent controlled
experiment tests full MIRA, zero explicit-mask input, zero distance input, and
uniform pooling. All four retain the same 20-channel architecture and 40,871
nominal parameters. A disabled channel has zero input and no data-dependent
weight gradient; nominal parameter matching does not make the active information
or effective capacity identical. Under the defined nearest-support distance,
**`m_t = I[d_t = 0]`**: distance fully determines the mask, while distance is also
a deterministic function of the complete mask. The zero-mask-input control tests
the explicit encoding rather than removal of missingness information; distance
and mask-dependent coverage pooling remain available.

Two conventional residual 1D CNN controls, trained clean or with loss augmentation,
each have 511,111 parameters. They are reference architectures, not new MIRA
components.

## Evaluation stages

| Stage | Scope | Principal result |
|---|---|---|
| Released v0.1.0 comparison | 60 fits: four models × three folds × five seeds | MIRA: complete 23.73 ± 2.26%; B50 linear 23.00 ± 1.28% under the five paired-seed summary |
| Frozen component controls | 90 fits: six configurations × three folds × five seeds; 31 conditions per fit | Full MIRA: complete 23.73 ± 2.26%; R50 linear 22.68 ± 2.03%; B50 linear 22.56 ± 1.28% under the full mask grid |
| Clean MiniRocket representation sensitivity | 30 reference/selected fits; five seeds and three folds | Official MiniRocket: original 53.08 ± 0.33%; source-selected 57.44 ± 1.04% |

Accuracy values are means ± sample SD. In the follow-up, each training/mask cell
first pools all 1,260 held files. Five mask scores are averaged within a training
seed; the mean and sample SD then use five training seeds. Macro F1 is computed
from each pooled confusion matrix, not averaged from fold F1 scores. The initial
**23.00%** block result uses one paired mask per training seed; **22.56%** uses
all five masks for every training seed. They summarize different retained views
of the same recordings, not different datasets or a model replacement.

The 90-fit follow-up preserves 5,580 metric rows and 1,640,520 probability vectors
over familiar and held sets. These are repeated views, not additional people.
Independent arithmetic reconstruction reproduced every row to maximum absolute
error 4.44×10⁻¹⁶. A CPU audit verified 90 source checkpoint hashes and reproduced
15,120 source-validation prediction views with no class-label mismatch; its
maximum CPU/T4 logit difference was 6.20×10⁻⁶. The Colab post-evaluation audit
reported unchanged checkpoint hashes. Post-evaluation weights were not reloaded
locally during the probability-only audit.

### Component findings

The exploratory B50-linear macro-F1 family has three full-minus-control contrasts:

| Control | Difference | Conditional 95% range | Bonferroni 98.333% range |
|---|---:|---:|---:|
| Zero explicit mask input | +0.0039 | [−0.0074, +0.0155] | [−0.0096, +0.0180] |
| Zero distance input | −0.0047 | [−0.0080, −0.0015] | [−0.0086, −0.0007] |
| Uniform temporal pooling | +0.0137 | [+0.0001, +0.0275] | [−0.0023, +0.0302] |

The 10,000 paired bootstrap draws resample ten participants with replacement
within each of three fixed environments. Each sampled person retains the full
seed/mask grid; all configurations share the same draws. The explicit mask-input
contrast is inconclusive here. The fitted-grid mean slightly favours removing
distance. The apparent pooling benefit does not survive the three-comparison
adjustment. These participant intervals condition on the already fitted seed grid;
they do not describe the distribution of effects under retraining.

| Control | Paired F1 difference, mean ± seed SD | Seeds favouring full MIRA |
|---|---:|---:|
| Zero explicit mask input | +0.0039 ± 0.0126 | 3/5 |
| Zero distance input | −0.0047 ± 0.0117 | 2/5 |
| Uniform temporal pooling | +0.0137 ± 0.0190 | 4/5 |

Each difference is full minus control at the same training seed after averaging
the five B50-linear masks. These descriptive five-seed SDs and sign counts reveal
mixed directions; they are not confidence intervals. In particular, the adjusted
participant range for distance does not imply that removing distance helps at
every seed or under future retraining. The results do not establish individual,
retraining-stable benefits for all additions. See the
[15 paired differences](evaluation/component-controls/summaries/component-differences-per-seed.csv)
and [seed-effect summary](evaluation/component-controls/summaries/component-differences-seed-summary.csv),
derived from the saved mask-averaged seed scores without new fitting.

### MiniRocket sensitivity is a separate comparator result

Five representations were fixed before fitting. Each outer fold selects a
representation using only that fold's source-validation macro F1 at the first
training seed, and then keeps it fixed across five seeds. E1 selects temporal
per-band standardization; E2/E3 select scalar per-recording standardization.
Validation scores are not pooled across folds for selection. All 30 states were
sealed before held scoring; six first-seed fits were reused and 24 more were fitted.

The selected comparator reaches 57.44% overall clean accuracy. Per-environment
means change from 57.24% to 66.43% in E1, 50.67% to 50.43% in E2 and 51.33% to
55.48% in E3. The gain is not uniform. This is **MiniRocket**, not a MIRA result.
This stage uses complete recordings and does not establish missing-input
robustness or leakage-safe standardization when values are hidden.

## Practical limits and evidence scope

- Room and participant cohort change together; a causal room-only effect cannot
  be identified from these splits.
- The follow-up was designed after earlier results on these data were observed.
  Its frozen execution is exploratory, not an untouched external confirmation.
- Synthetic removal of normalized packet positions does not cover all packet
  acquisition failures, sampling irregularities or real deployment conditions.
- Source-only temperatures need not remain calibrated after transfer or loss.
- Seed SD measures training variation conditional on fixed data and partitions.
  Participant-bootstrap ranges condition on fixed fitted models, recordings,
  rooms and masks; they exclude new-room and retraining uncertainty.
- Absolute recognition performance remains limited. These results do not establish
  general superiority, state-of-the-art performance or deployment readiness.

[Initial evaluation](evaluation/README.md) ·
[Follow-up summaries, inference scope and real plots](evaluation/component-controls/README.md) ·
[Saved-model quick start](README.md#quick-start)
