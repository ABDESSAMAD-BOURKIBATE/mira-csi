# MIRA-CSI model card

**Developer and research lead:** Eng. Abdessamad Bourkibate<br>
**Task:** offline classification of recorded Wi-Fi CSI into seven activity labels<br>
**Status:** experimental software; the runnable release is v0.1.0.

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
