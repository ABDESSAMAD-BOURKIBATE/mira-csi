# MIRA-CSI spectral variant E

**Experimental offline Wi-Fi CSI activity classification**

Developed by **Abdessamad Bourkibate**. This variant combines observed-endpoint
lagged-increment generalized spectra with a compact, 25,415-parameter MLP.
It classifies seven activity labels from an existing **18 × 500 log-mean-power
CSI representation**. The package provides the frozen implementation, 15
trained E checkpoints, source-validation calibration, a standalone predictor,
and all-arm comparison tables.

This is a follow-up to the MIRA-CSI project, not MiniRocket under a new name.
Covariance spectra, temporal normalization, masking, CNNs and MLPs have prior
art. The implemented endpoint construction and its comparison are attributed
and described in [MODEL-CARD.md](MODEL-CARD.md) and [NOTICE.md](NOTICE.md).

## Run the included recording

Use Python 3.12 and an environment with the dependencies in `requirements.txt`.
The tested CPU environment is recorded in `runtime-tested.json`. For a fresh
environment, install dependencies using the usual package manager, then:

```bash
python tools/verify_release.py
python predict_spectral_e.py --input demo/first-held-E1-power18.npy
python predict_spectral_e.py --input demo/first-held-E1-power18.npy --mask demo/first-held-E1-B50-mask.npy
python -m unittest discover -s tests -v
```

The default is fold E1, training seed 20261026. Select another recorded model
with `--fold 2 --seed 20261027`. These choices identify separate trained source
models, not a target-guided automatic selector or an ensemble. Add
`--output prediction.json` to save the class probabilities.

The demo is **the first E1 held recording in the immutable manifest**, not an
example selected for success. Its true label is `sit`; the saved predictions
are retained even though the model is wrong in both complete and B50 views.
`demo/stored-reference.json` contains the original batched outputs;
`demo/cli-*.json` contains the standalone replays. The agreement report gives
actual numeric differences and the fixed float32 tolerances. Matching the
earlier output validates this CLI path; it does not make the prediction correct.

## Supply your own processed CSI

The input is a real numeric `.npy` array of shape `(18, 500)`, **before any
source-training mean/std normalization**. Each stream is the fixed
`10*log10(max(mean(abs(H_scaled)**2), 1e-24))` over a group of five exported
subcarriers, for three receive antennas and six groups per antenna, resampled
to 500 normalized valid-packet positions. See [DATA-ATTRIBUTION.md](DATA-ATTRIBUTION.md).
These are dimensionless scaled-power dB values, not calibrated dBm.

An optional mask must have shape `(500,)`, with Boolean or finite numeric
entries exactly 0 or 1; 1 means observed. NaN, Inf, nonbinary and incorrectly
shaped masks are rejected **before** Boolean conversion. Observed input values
must be finite. Hidden numeric NaN/Inf fills are ignored. An entirely empty
record is rejected because complete loss was not evaluated. E uses the same
actual-mask feature extractor as arm A, with no temporal CNN or imputer.
Numerical overflow or eigensolver failure, non-finite features/logits and
non-finite or invalid probabilities are rejected before returning results or
writing JSON. Frozen modules are loaded from the same package root whose
bytes were verified, without reusing unqualified module-cache names.

An ordinary Wi-Fi connection is not this input. This package does not capture
CSI from an unsupported adapter, reconstruct a body image, estimate vital
signs or validate continuous live sensing. It needs real CSI measurements or
the provided existing recordings; it cannot show a person from their router
connection alone.

## What was measured

The selected dataset has 1,260 recordings from 30 people in three different
environment/cohort groups. Each outer fold uses 504 source-training, 168
source-validation, 168 familiar-test and 420 held-environment recordings.
Five arms A--E were trained across three folds and five seeds: 75 fits.
All weights and source-calibration values were frozen before held/familiar
model inference in this execution. Static recordwise preprocessing and hash
verification are distinct from inference or target fitting. Historical target
results had already influenced the research direction, so this remains an
exploratory extension, not fresh preregistered confirmation.

Pooled held results for E, **mean ± sample SD over five training seeds after
averaging five masks within each seed**:

| Input | Accuracy (%) | Macro F1 (%) | NLL | Brier |
|---|---:|---:|---:|---:|
| Complete | 75.175 ± 1.005 | 74.607 ± 1.207 | 0.773 ± 0.018 | 0.365 ± 0.008 |
| Random 50% loss | 73.822 ± 0.942 | 73.329 ± 1.105 | 0.831 ± 0.018 | 0.386 ± 0.009 |
| Contiguous 50% loss | 60.597 ± 1.375 | 59.159 ± 1.610 | 1.160 ± 0.014 | 0.552 ± 0.011 |

The primary hybrid A did substantially worse: 48.302% complete and 34.775%
contiguous-loss accuracy. Its primary A minus B contiguous-loss macro-F1
contrast is only **+0.543 ± 0.654 percentage points**, with four positive and
one negative training seed. E was prespecified as a comparator but became
prominent after seeing these outcomes. No feature-only B-denominator control
was tested, so E's result does not isolate a benefit caused by support matching.

The 90-row fold/domain table retains all A--E results. Familiar E B50 accuracies
are 56.024%, 48.524%, 55.429% in outer folds E1/E2/E3. Two pooled E training
seeds also remain below 60% under B50. Neither every run nor B50 macro F1
exceeds 60%. SD over five seeds is not a population confidence interval.
There is no controlled superiority claim over a differently processed
MiniRocket benchmark, no SOTA claim, and no certificate of scientific priority.

## Reproduce or inspect the frozen study

`frozen/` contains the eight original implementation/test files and the exact
protocol bytes. Their historical comments describe the architecture at its
candidate-definition stage; the trained states are the separate `artifacts/`.
Only the 15 E checkpoints are included here. `artifacts/source_fit_freeze.json`
records the original all-75 freeze; it does not imply that A--D weights are in
this E package. No optimizer or resume states are distributed.

An optional separate training-input ZIP contains the fixed 45 MB derived
power cache and manifest under the dataset's CC BY 4.0 terms. Extract its
contents to `data/` to reproduce training, then use a fresh output directory:

```bash
python frozen/run_training.py --data-root data --output-root reproduced
```

The default is source-only training of all 75 fits. Evaluation is a separate
stage and the frozen runner refuses it until all 75 new source fits are
complete and frozen together. The supplied 15 E files do not bypass that gate.
The saved small reports permit inspection without repeating training.

## File integrity and licences

`tools/verify_release.py` checks the declared file hashes, all 15 E weight and
validation bindings, and recomputes saved source-validation metrics without
model inference. The predictor pins the protocol and all-fit freeze hashes,
checks checkpoint/validation/training-receipt hashes before Torch loading,
and checks class order and temperature against that frozen record.
Its restricted weights-only loader allows only the NumPy types required by
the original checkpoints; it has no unsafe pickle-loader fallback.
The two saved real-record CLI replays precede the subsequent finite-value and
verified-root hardening; their interface hash is retained in the agreement
record. The hardened boundary was checked with synthetic/mocked outputs,
without repeating real-record model inference. Frozen features and weights
remain unchanged.

**Hashes establish consistency with the recorded files, not publisher
authenticity, endorsement or external validity.** They are not signatures.

Software: GPLv3, retaining the existing MIRA-CSI licence. The original dataset
and included derived recording remain attributed separately under CC BY 4.0.
Dependency software keeps its own licence. No private media, credentials,
review prompts, workstation paths or runtime launch files are part of the
public package. See `LICENSE`, [NOTICE.md](NOTICE.md),
[DATA-ATTRIBUTION.md](DATA-ATTRIBUTION.md) and `licenses/`.

For attribution of this implementation, use [CITATION.cff](CITATION.cff).
Please also cite the original dataset and relevant methodological sources;
citing this package does not replace those citations.
