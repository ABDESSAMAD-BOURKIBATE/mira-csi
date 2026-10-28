<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cover-dark-transparent.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/cover-light-transparent.png">
    <img src="docs/assets/cover-light-transparent.png" alt="MIRA-CSI — Missingness-Informed Wi-Fi Activity Recognition, Eng. Abdessamad Bourkibate" width="900">
  </picture>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12_%7C_3.13-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-numerics-4DABCF?style=flat-square&logo=numpy&logoColor=white">
  <img alt="Numba" src="https://img.shields.io/badge/Numba-MiniRocket-00A3E0?style=flat-square&logo=numba&logoColor=white">
  <img alt="LaTeX documentation" src="https://img.shields.io/badge/LaTeX-documentation-008080?style=flat-square&logo=latex&logoColor=white">
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/ABDESSAMAD-BOURKIBATE/mira-csi/blob/main/notebooks/quickstart.ipynb"><img alt="Open quick start in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.3.0"><img alt="Research version" src="https://img.shields.io/badge/research_version-v0.3.0-176B74?style=flat-square&logo=github"></a>
  <img alt="Frozen component controls" src="https://img.shields.io/badge/follow--up-90_fits_verified-14806A?style=flat-square">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-GPL--3.0-6C5B96?style=flat-square"></a>
  <a href="https://orcid.org/0009-0000-6186-8071"><img alt="Author ORCID" src="https://img.shields.io/badge/ORCID-author-A6CE39?style=flat-square&logo=orcid&logoColor=white"></a>
</p>

<p align="center"><a href="paper/README.md">Research paper</a> · <a href="#quick-start">Quick start</a> · <a href="#architecture">Architecture</a> · <a href="#evaluation">Verified results</a> · <a href="#repository-map">Repository map</a> · <a href="MODEL_CARD.md">Model card</a> · <a href="#citation">Citation</a></p>

**New in v0.3.0:** [`spectral/`](spectral/) contains **MIRA-CSI spectral variant E (experimental)**, a feature-only classifier with 15 saved E checkpoints, source calibration and a checked standalone predictor. Start with the spectral commands below. The root [`predict.py`](predict.py) and existing Colab notebook retain the **original temporal model** and its different amplitude representation; the two predictors and their checkpoints are not interchangeable.

<a id="overview"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/overview-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/overview-light-v2.svg">
  <img src="docs/assets/headings/overview-light-v2.svg" alt="Overview" width="900">
</picture></h3>

**MIRA-CSI** — Missingness-Informed Recognition of Activities from Wi-Fi CSI — is a PyTorch research project for offline activity classification from incomplete CSI recordings. It preserves the original temporal model and component controls, and adds **spectral variant E**: observed-endpoint lagged-increment features followed by a compact MLP. The repository provides runnable saved-model predictors, frozen execution records and results that include unsuccessful controls.

**Developer and research lead:** Eng. **Abdessamad Bourkibate**  
Software & Cybersecurity Engineer | AI & EdTech — **Share'In Holding**

**Status:** experimental, **unreviewed research software; no assigned DOI**. The new spectral study contains 75 A–E fits; a subsequent fixed-power MiniRocket reference contains 15 configurations, comprising three reused source fits and 12 additional fits. E reaches **75.175 ± 1.005%** held accuracy on complete input and **60.597 ± 1.375%** under contiguous 50% loss. Its B50 macro F1 is **59.159 ± 1.610%**. These are exploratory, fixed-data results, not a deployment guarantee, SOTA ranking or patent claim. The original 60-fit comparison, separate 90-fit component study and clean-only MiniRocket representation sensitivity remain available below.

**Read the study:** [complete research manuscript — PDF](paper/manuscript.pdf) · [four-page supplementary results](paper/supplementary-results.pdf) · [source and page previews](paper/README.md). The current author version has **32 pages, 19 tables, 18 figures and 25 references**. It is not journal peer reviewed, has no assigned DOI and retains the limitations of the earlier studies. The [original component analysis](studies/component_removal/README.md) remains separately reproducible.

**v0.3.0 evidence assets:** [completed A–E study, validated](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-completed-A-E-spectral-evidence-20261028-validated.zip) · [fixed-power MiniRocket reference](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-matched-power-MiniRocket-evidence-20261028.zip) · [standalone E package](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-spectral-E-package.zip) · [E weights/calibration](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-spectral-E-weights.zip) · [separate power18 training inputs](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-spectral-training-inputs.zip). Each evidence bundle retains its protocol, hashes, numerical results and scope; the three stages must not be counted as independent new datasets.

**Download the evidence:** [v0.2.0 research release](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.2.0) · [validated reproducibility supplement](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.2.0/MIRA-CSI-reproducibility-supplement-20261028-validated.zip) · [SHA256 checksums](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.2.0/SHA256SUMS.txt). The original v0.1.0 predictor package remains linked in Quick start.

**Software update:** [v0.2.1 integrity patch](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.2.1) validates artifact reuse, checkpoint-bound calibration and portable text hashes. It preserves the recorded model results; [verification details](evaluation/software-integrity/README.md) include the clean-checkout check.

<details>
<summary><strong>Preview actual manuscript pages</strong></summary>

<p align="center">
  <a href="paper/manuscript.pdf"><img src="paper/first-page.png" alt="Actual first page of the complete research manuscript" width="340"></a>
  <a href="paper/manuscript.pdf"><img src="paper/component-page.png" alt="Actual manuscript page with six-model score plots and paired component contrasts" width="340"></a>
</p>

The first page is from the current author version; the component-page preview is retained from the earlier author version. Current spectral and reference pages appear below. See [paper/MANIFEST.json](paper/MANIFEST.json) for the versioned PDF and source hashes.

<p align="center">
  <a href="paper/manuscript.pdf"><img src="paper/spectral-results-page.png" alt="Current manuscript page with spectral A–E results" width="340"></a>
  <a href="paper/manuscript.pdf"><img src="paper/matched-reference-page.png" alt="Current manuscript page with the fixed-power MiniRocket reference" width="340"></a>
</p>

</details>

<a id="quick-start"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/quick-start-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/quick-start-light-v2.svg">
  <img src="docs/assets/headings/quick-start-light-v2.svg" alt="Quick start" width="900">
</picture></h3>

**Spectral variant E — current saved-model entry point**

Run from the repository root with Python 3.12 and the dependencies in [`spectral/requirements.txt`](spectral/requirements.txt):

```sh
python -m pip install -r spectral/requirements.txt
python spectral/tools/verify_release.py
python spectral/predict_spectral_e.py --input spectral/demo/first-held-E1-power18.npy
python spectral/predict_spectral_e.py --input spectral/demo/first-held-E1-power18.npy --mask spectral/demo/first-held-E1-B50-mask.npy
python -m unittest discover -s spectral/tests -v
```

The predictor consumes an **unnormalized processed log-mean-power18** array of shape `(18, 500)`, not raw complex CSI or the original predictor's amplitude tensor. Its optional `(500,)` mask is Boolean or finite binary, with 1 meaning observed. It rejects invalid masks before Boolean conversion, non-finite observed values, numerical feature failures and non-finite outputs. The package binds 15 E checkpoints, class order and source temperatures to the recorded freeze; hashes establish file consistency, not authenticity. [Full input contract and tested environment](spectral/README.md).

The demo is the first E1 held recording in manifest order, not a successful example selected for display: its true label is `sit`, while the model predicts `pick_up` on complete input and `stand` under the included block mask. Both errors are retained. The saved replay agrees with the stored batched probabilities within the declared float32 tolerance; subsequent interface hardening was tested without additional real-record inference.

**Original temporal predictor — retained historical entry point**

The [executed saved-model notebook](notebooks/quickstart.ipynb) and commands below use the original v0.1.0 temporal model. The notebook has been run top-to-bottom on a Windows CPU and retains its actual outputs. Use its own checkpoint, calibration and amplitude input together; do not substitute spectral files.

```sh
git clone https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi.git
cd mira-csi
python -m pip install numpy
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Download [the verified model/results package](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.1.0/mira-csi-verified-results.zip) and extract it into the project directory, producing `results/`. Then run:

```sh
python predict.py --checkpoint results/checkpoints/mira_csi_E3_s20261026.pt --input results/demo_input.npy --calibration results/runs/mira_csi_E3_s20261026.json
```

Add `--observed results/demo_observed.npy` for the matched block-loss example. A CPU is sufficient for saved-model inference. The predictor returns an activity label and seven model probabilities. The included example is a public recorded CSI segment.

Exact evaluated package versions are in [`evaluation/package_versions.json`](evaluation/package_versions.json).
The current predictor verifies exact released checkpoint/calibration pairs through a small [release manifest](evaluation/legacy-release-manifest.json). New runs bind calibration to checkpoint bytes; unknown historical pairs require an explicit legacy opt-in.
See the [software integrity checks](evaluation/software-integrity/README.md) for corruption tests and the limits of these checks.

<a id="architecture"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/architecture-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/architecture-light-v2.svg">
  <img src="docs/assets/headings/architecture-light-v2.svg" alt="Architecture" width="900">
</picture></h3>

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/architecture-light.png">
  <img src="docs/assets/architecture-light.png" alt="18 amplitude bands plus observation mask and gap channels, temporal CNN, coverage-weighted pooling and seven activity logits" width="900">
</picture></p>

**The diagram above and the following table describe the original temporal model.** Spectral E instead extracts 360 actual-mask features over whole/half-record cells and six lags, then uses a 360 → 64 → 32 MLP and seven-class head, with **25,415 parameters and no temporal CNN**. A–C are the corresponding hybrid controls; D is temporal-only. Covariance spectra and temporal normalization have prior art in SFA/SOBI, and PADS is earlier CSI covariance/eigen-feature work. [Spectral model and attribution](spectral/MODEL-CARD.md) · [primary methodological sources](spectral/NOTICE.md).

| Component | Implementation |
|---|---|
| Signal input | 18 amplitude bands × 500 normalized packet positions |
| Missingness input | Observation mask and distance to nearest observation |
| Encoder | 1D convolutions: 32 → 64 → 128 channels, GroupNorm, GELU |
| Pooling | Temporal features weighted by local observation coverage |
| Output | Seven logits, with source-validation temperature scaling |
| Model size | 40,871 trainable parameters |

The initial comparison includes an ordinary amplitude CNN (40,423 parameters), the same CNN with source-only loss augmentation, and the official MiniRocket transform with **9,996 features**. The follow-up tests zero explicit-mask input, zero distance input and uniform pooling in otherwise matched compact networks, plus two conventional residual CNN controls. Under the defined nearest-support distance, **`m_t = I[d_t = 0]`**: the mask is fully recoverable from the distance channel. Zero mask input therefore tests an explicit encoding, rather than removal of missingness information; mask-dependent coverage pooling also remains available. Mask, gap and learned missing-CSI recovery methods have prior art; [`PRIOR-ART.md`](PRIOR-ART.md) documents attribution and the limits of the novelty claim.

<a id="measurement-availability-case-by-case"></a>
<h4><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/measurement-availability-case-by-case-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/measurement-availability-case-by-case-light-v2.svg">
  <img src="docs/assets/headings/measurement-availability-case-by-case-light-v2.svg" alt="Measurement availability, case by case" width="900">
</picture></h4>

The first three clips are abstract data-flow illustrations. Teal paths represent parallel streams and dots represent observations; spacing and motion do not encode measured amplitudes, time, radio propagation or a numerical loss rate. The final plot uses the released real CSI example.

<a id="complete-observations"></a>
<h4><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/complete-observations-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/complete-observations-light-v2.svg">
  <img src="docs/assets/headings/complete-observations-light-v2.svg" alt="Complete observations" width="900">
</picture></h4>

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/case-complete-dark.webp">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/case-complete-light.webp">
  <img src="docs/assets/case-complete-light.webp" alt="Conceptual complete observation streams" width="640">
</picture></p>

Available samples enter preprocessing together. Complete input means no additional mask was imposed; it does not establish that acquisition was free from noise or packet loss.

<a id="random-missing-positions"></a>
<h4><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/random-missing-positions-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/random-missing-positions-light-v2.svg">
  <img src="docs/assets/headings/random-missing-positions-light-v2.svg" alt="Random missing positions" width="900">
</picture></h4>

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/case-random-loss-dark.webp">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/case-random-loss-light.webp">
  <img src="docs/assets/case-random-loss-light.webp" alt="Conceptual scattered missing observations" width="640">
</picture></p>

Scattered gaps illustrate non-contiguous unavailable positions. In the experiment, the exact requested number of positions is selected without replacement and the same temporal mask is applied to all 18 streams.

<a id="contiguous-missing-interval"></a>
<h4><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/contiguous-missing-interval-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/contiguous-missing-interval-light-v2.svg">
  <img src="docs/assets/headings/contiguous-missing-interval-light-v2.svg" alt="Contiguous missing interval" width="900">
</picture></h4>

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/case-block-loss-dark.webp">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/case-block-loss-light.webp">
  <img src="docs/assets/case-block-loss-light.webp" alt="Conceptual contiguous observation gap" width="640">
</picture></p>

A common gap illustrates a continuous unavailable interval. In the evaluated 50% case, exactly 250 of 500 normalized packet positions are hidden jointly across streams. The illustration itself does not encode this fraction.

<a id="repair-on-the-actual-recording"></a>
<h4><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/repair-on-the-actual-recording-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/repair-on-the-actual-recording-light-v2.svg">
  <img src="docs/assets/headings/repair-on-the-actual-recording-light-v2.svg" alt="Repair on the actual recording" width="900">
</picture></h4>

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/case-linear-repair-dark.webp">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/case-linear-repair-light.webp">
  <img src="docs/assets/case-linear-repair-light.webp" alt="Measured CSI band with retained observations and observed-only linear repair" width="640">
</picture></p>

Grey is the original public recording, teal is retained input, and dashed amber is linear interpolation between observed boundaries. The grey hidden values are used only for reconstruction scoring, never for repair. These offline operators can use later observed samples. Normalized packet positions are not elapsed seconds. [Visual provenance](docs/assets/CASE-VISUAL-PROVENANCE.json).

<a id="evaluation"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/evaluation-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/evaluation-light-v2.svg">
  <img src="docs/assets/headings/evaluation-light-v2.svg" alt="Evaluation" width="900">
</picture></h3>

<h4>Spectral E and the fixed-power MiniRocket reference · v0.3.0</h4>

The 75-fit A–E study evaluates three held-environment folds and five training seeds. A further MiniRocket reference uses the **same power18 representation, participant roles, seeds and test masks**: three source fits were reused and 12 new source fits completed its 15 configurations. E uses loss augmentation and actual-mask features; this MiniRocket is **clean-trained, with source-band-mean fill and no mask input**. The comparison therefore includes training and loss-treatment differences and does not isolate architecture or support matching.

| Model | Input | Held accuracy (%) | Macro F1 (%) | NLL | Brier |
|---|---|---:|---:|---:|---:|
| Spectral E | Complete | 75.175 ± 1.005 | 74.607 ± 1.207 | 0.773 ± 0.018 | 0.365 ± 0.008 |
| Spectral E | Random 50% | 73.822 ± 0.942 | 73.329 ± 1.105 | 0.831 ± 0.018 | 0.386 ± 0.009 |
| Spectral E | Contiguous 50% | 60.597 ± 1.375 | 59.159 ± 1.610 | 1.160 ± 0.014 | 0.552 ± 0.011 |
| MiniRocket power18 | Complete | 53.841 ± 0.938 | 53.868 ± 1.009 | 1.363 ± 0.037 | 0.634 ± 0.009 |
| MiniRocket power18 | Random 50% | 22.305 ± 1.102 | 16.112 ± 0.841 | 3.486 ± 0.151 | 1.109 ± 0.023 |
| MiniRocket power18 | Contiguous 50% | 19.406 ± 2.330 | 15.636 ± 4.147 | 4.852 ± 1.764 | 1.275 ± 0.168 |

Each training/mask cell pools the three held folds before scoring. Masks are averaged within a training seed; entries show **mean ± sample SD over five training seeds**, not population confidence intervals. Brier is summed over seven classes. Complete input has one realization; each loss condition has five masks.

The hybrid A/B controls remain weaker than this reference on complete held input: **48.302 ± 2.627%** and **48.048 ± 2.559%** accuracy. The original primary A−B B50 macro-F1 contrast is only **+0.543 ± 0.654 percentage points**, with four positive and one negative training seed. It changes endpoint weighting and centering even on complete data; E's feature-only counterpart with B's denominator was not tested.

The additional, **post-hoc** E−MiniRocket B50 macro-F1 contrast is **+43.524 ± 3.639 percentage points**. Its 10,000-draw participant-bootstrap range, **[39.879, 47.054]**, is descriptive and conditional on the fixed rooms, fitted models, seeds and masks; it is not selection-corrected confirmation or an uncertainty interval for new rooms/training datasets. Historical target outcomes were already known during development. E's B50 pooled familiar accuracy is **53.325 ± 1.480%**, and two pooled held seed accuracies are below 60%. The results do not establish a universally successful classifier, a matching mechanism, SOTA or patent novelty. [All A–E results](spectral/results/pooled_held_mean_sampleSD_5seeds.csv) · [full fold/domain supplement](paper/supplementary-results.pdf) · [reference evidence asset](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-matched-power-MiniRocket-evidence-20261028.zip).

**Earlier temporal-model study (v0.2.0):** the following plots, headings and results are retained as historical evidence; they describe the original CNN, not spectral E.

<h4>Frozen follow-up · six configurations on the same recordings</h4>

**90 fits · 3 excluded-domain folds · 5 training seeds · 31 conditions per fit**

The follow-up retains **5,580 metric rows** and **1,640,520 probability vectors** across familiar and held sets. These are repeated views of **1,260 recordings from 30 participants**, not additional people. Checkpoints and source-validation temperatures were frozen before scoring. Room and participant cohort change together, so these splits cannot isolate a causal room-only effect.

| Configuration | Complete accuracy (%) | Random 50%, linear (%) | Block 50%, linear (%) |
|---|---:|---:|---:|
| Full MIRA | 23.73 ± 2.26 | 22.68 ± 2.03 | 22.56 ± 1.28 |
| Zero explicit mask input | 23.41 ± 1.03 | 23.22 ± 1.23 | 21.90 ± 0.53 |
| Zero distance input | 24.03 ± 1.87 | 22.85 ± 1.87 | 22.77 ± 1.02 |
| Uniform temporal pooling | 24.17 ± 2.54 | 23.22 ± 1.86 | 21.30 ± 1.57 |
| Residual CNN, clean training | 33.83 ± 0.82 | 33.33 ± 0.87 | 26.72 ± 0.67 |
| Residual CNN, augmented | 30.25 ± 1.65 | 30.09 ± 1.70 | 25.72 ± 1.35 |

Each training/mask cell first pools the three held sets into 1,260 predictions. Corrupted-case scores are averaged over five masks within each training seed; cells show the mean ± sample SD across five training seeds. F1 uses each pooled confusion matrix, not the mean of fold F1. The four compact configurations retain **40,871 nominal parameters**; each residual control has **511,111**.

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/followup-models-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/followup-models-light.png">
  <img src="docs/assets/followup-models-light.png" alt="Real held results for all six frozen configurations: accuracy, macro F1 and source-calibrated NLL, with five-seed SD" width="900">
</picture></p>

C denotes complete input; R50/B50 denote random/contiguous 50% loss with linear repair. Error bars show sample SD across training seeds. Higher accuracy/F1 and lower NLL are preferred. The plots are rendered directly from the saved numerical results.

**Why the earlier block result was 23.00%:** the original v0.1.0 primary summary uses five paired training/mask seeds. The follow-up's **22.56%** averages the full five-by-five training/mask grid. Both use the same selected recordings; the change is the aggregation grid. The original release remains available below and its predictor is unchanged.

<h4>What the component controls establish</h4>

The three exploratory contrasts are full MIRA minus each removal control in B50-linear macro F1. All use the same **10,000 participant-cluster bootstrap draws**, sampling ten people within each fixed environment. Positive differences favour full MIRA; only this three-comparison family receives the stated Bonferroni adjustment.

| Control | Δ Macro F1 | Conditional 95% | Bonferroni 98.333% |
|---|---:|---:|---:|
| Zero explicit mask input | +0.0039 | [−0.0074, +0.0155] | [−0.0096, +0.0180] |
| Zero distance input | −0.0047 | [−0.0080, −0.0015] | [−0.0086, −0.0007] |
| Uniform temporal pooling | +0.0137 | [+0.0001, +0.0275] | [−0.0023, +0.0302] |

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/followup-components-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/followup-components-light.png">
  <img src="docs/assets/followup-components-light.png" alt="Three actual component F1 contrasts, showing unadjusted 95 percent and adjusted 98.333 percent conditional participant-bootstrap ranges" width="900">
</picture></p>

The fitted-grid mean slightly favours removing distance in this condition. The explicit-mask contrast is inconclusive, and the pooling benefit does not survive the family correction. These participant-bootstrap ranges condition on the existing fitted models, recordings, rooms and masks; **they do not cover retraining uncertainty**. A family-adjusted interval excluding zero is therefore not a guarantee that the effect has the same direction for each training seed or a future fit.

| Control | Paired F1 difference, mean ± seed SD | Seeds favouring full MIRA |
|---|---:|---:|
| Zero explicit mask input | +0.0039 ± 0.0126 | 3/5 |
| Zero distance input | −0.0047 ± 0.0117 | 2/5 |
| Uniform temporal pooling | +0.0137 ± 0.0190 | 4/5 |

These descriptive differences subtract each control's F1 from full MIRA **within the same training seed, after averaging all five masks**. Their signs vary across the fitted seeds. Computed from the saved [mask-averaged seed scores](evaluation/component-controls/summaries/held-mask-averaged-per-seed.csv); [all 15 paired differences](evaluation/component-controls/summaries/component-differences-per-seed.csv) and their [three-row summary](evaluation/component-controls/summaries/component-differences-seed-summary.csv) retain the calculation. The evidence does **not** establish an independently beneficial or retraining-stable effect for every MIRA addition. [All repairs and inference scope](evaluation/component-controls/README.md).

<h4>A separate clean-input MiniRocket result</h4>

Source-only, fold-specific representation selection raises the official MiniRocket comparator from **53.08 ± 0.33%** to **57.44 ± 1.04%** clean held accuracy. Each outer fold chooses among five fixed representations using only its own source-validation macro F1, then retains that representation across five seeds. Validation scores are never pooled across folds for selection.

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/followup-minirocket-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/followup-minirocket-light.png">
  <img src="docs/assets/followup-minirocket-light.png" alt="Real clean MiniRocket results for E1, E2, E3 and the pooled held set; original versus source-selected representations" width="900">
</picture></p>

All environments are shown: E1 changes from 57.24% to 66.43%, E2 from 50.67% to 50.43%, and E3 from 51.33% to 55.48%. The improvement is not uniform. **This is MiniRocket's result, not MIRA's.** Full-recording normalization in this clean-only study establishes neither missing-input robustness nor a safe normalization procedure when values are hidden. [Exact results and selection rules](evaluation/component-controls/README.md#separate-clean-input-minirocket-sensitivity).

<details>
<summary><strong>Original v0.1.0 comparison · 60 fits · five paired training/mask seeds</strong></summary>

| Model | Complete accuracy (%) | Block 50%, linear (%) |
|---|---:|---:|
| CNN | 24.52 ± 2.17 | 22.19 ± 2.15 |
| CNN + augmentation | 22.54 ± 1.82 | 20.76 ± 1.33 |
| MIRA-CSI | 23.73 ± 2.26 | 23.00 ± 1.28 |
| MiniRocket (9,996) | 53.08 ± 0.33 | 19.59 ± 0.88 |

These means ± sample SD use five paired training/mask seeds; each seed pools 1,260 held files. This is the original summary, not the full mask-grid component comparison above.

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/comparison-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/comparison-light.png">
  <img src="docs/assets/comparison-light.png" alt="Original v0.1.0 familiar and held-environment results under complete input and imposed loss" width="900">
</picture></p>

The [original evaluation](evaluation/README.md) includes 3,048 independently reconstructed metric rows and 896,112 saved probability vectors, including dependent supplementary mask views. It retains per-seed metrics, repairs, NLL, Brier, ECE and a failed first-file inference example. Those counts describe v0.1.0, not the new 90-fit follow-up.

</details>

The follow-up's 5,580 scores were independently reconstructed with maximum absolute difference 4.44×10⁻¹⁶. A separate CPU audit verified all 90 source checkpoint hashes and reproduced 15,120 source-validation prediction views without a class-label mismatch. The [audit summary](evaluation/component-controls/summaries/verification-summary.json) distinguishes local source-weight verification from the Colab post-evaluation hash check. None of these checks establishes deployment readiness or broad model superiority.

<a id="technology-stack"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/technology-stack-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/technology-stack-light-v2.svg">
  <img src="docs/assets/headings/technology-stack-light-v2.svg" alt="Technology stack" width="900">
</picture></h3>

| Role | Languages and libraries |
|---|---|
| Model and inference | Python · PyTorch · NumPy |
| Signal preparation | Python · NumPy · SciPy |
| Reference comparison | Numba · scikit-learn · official MiniRocket source |
| Analysis and figures | pandas · Matplotlib · Pillow |
| Research documentation | Markdown · LaTeX · CFF citation metadata |

<a id="reproduce"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/reproduce-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/reproduce-light-v2.svg">
  <img src="docs/assets/headings/reproduce-light-v2.svg" alt="Reproduce" width="900">
</picture></h3>

**Spectral study:** the checked E predictor and exact frozen A–E source are in [`spectral/`](spectral/). For a fresh 75-fit source-training reproduction, extract the [separate power18 input asset](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-spectral-training-inputs.zip) into `spectral/`, so that its `data/` contains the cache and manifest. Then run:

```sh
python spectral/frozen/run_training.py --data-root spectral/data --output-root spectral-reproduction
```

The default trains source models only. The frozen runner permits its separate evaluation stage only after all 75 new source fits are completed and frozen; the supplied 15 E checkpoints do not bypass that gate. The validated A–E and MiniRocket evidence assets above retain the completed-study outputs. [Input lineage, commands and licences](spectral/README.md).

**Original temporal study:** install `requirements.txt`. Obtain its own processed public input cache from the [v0.1.0 release assets](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.1.0), or rebuild it from pinned raw recordings:

```sh
python -m pip install -r requirements.txt
python prepare_data.py
python check_inputs.py
python run_experiment.py --output-root results/reproduction --models cnn_clean cnn_augmented mira_csi
python run_experiment.py --output-root results/reproduction --models minirocket_9996
python verify_results.py --output-root results/reproduction
python summarize_results.py --output-root results/reproduction
```

Use a fresh `results/reproduction` directory for a new experiment. A completion file alone is insufficient for reuse: current runs check source/configuration, inputs, run identity, checkpoint, saved probabilities, scores and histories. Partial, changed or unsealed historical outputs are refused. The published v0.1.0 package remains available for prediction and independent recalculation; it is not retroactively marked as a new run.

[`requirements.txt`](requirements.txt) is the general installation list. For the original source-fit dependency versions, see [`requirements-original-evaluation.txt`](requirements-original-evaluation.txt) and the recorded [runtime versions](evaluation/package_versions.json); hardware can affect training. The separate component study has its own [pinned analysis requirements](studies/component_removal/requirements.txt).

[`protocol.json`](protocol.json) fixes the original selection, participant roles, model settings and seeds. The [compute amendment](protocol_amendment_01.json) pairs training/mask seeds for the original primary comparison. Extra CNN masks and a previously completed full-grid MiniRocket run remain supplementary sensitivity data. The separate [component-control protocol](studies/component_removal/protocol.json) records the frozen follow-up. [Follow-up CSVs and analysis instructions](evaluation/component-controls/README.md) keep its full-grid summaries distinct from v0.1.0.

<details>
<summary><strong>Download contents and provenance</strong></summary>

| Asset | Contents |
|---|---|
| `mira-csi-verified-results.zip` | All 60 trained states, probability arrays, metrics, histories, plots and run records |
| `mira-csi-public-inputs.zip` | Exact public input tensor, manifest, attribution and hashes |
| GitHub source archive | Source code, model card, evaluation summaries and documentation |

SHA256 digests accompany v0.1.0. Those release assets cover the original 60 fits; they do not contain the new 90-fit component checkpoints. The input cache does not overwrite current software source. Raw-data reconstruction downloads approximately 395 MB; the optional processed cache is smaller.

</details>

<a id="repository-map"></a>
<h4>Repository map</h4>

| Start here | Purpose |
|---|---|
| [`spectral/README.md`](spectral/README.md) · [`spectral/predict_spectral_e.py`](spectral/predict_spectral_e.py) | Current experimental spectral E, 15 saved weights, strict input checks and source-bound calibration |
| [`paper/supplementary-results.pdf`](paper/supplementary-results.pdf) | Four-page full A–E fold/domain/condition result tables |
| [`paper/manuscript.pdf`](paper/manuscript.pdf) · [`paper/source/`](paper/source/) | Complete author manuscript, actual page previews and matching LaTeX source |
| [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) · [`predict.py`](predict.py) | Executed v0.1.0 saved-model example and standalone inference |
| [`mira_model.py`](mira_model.py) · [`missingness.py`](missingness.py) | Released model and observed-only repair operators |
| [`protocol.json`](protocol.json) · [`protocol_amendment_01.json`](protocol_amendment_01.json) | Original experiment and paired-seed amendment |
| [`studies/component_removal/protocol.json`](studies/component_removal/protocol.json) | Separate frozen component/capacity experiment |
| [`evaluation/README.md`](evaluation/README.md) | Original 60-fit results, including failures |
| [`evaluation/component-controls/README.md`](evaluation/component-controls/README.md) | New 90-fit full-grid results, component contrasts and clean MiniRocket sensitivity |
| [`MODEL_CARD.md`](MODEL_CARD.md) · [`PRIOR-ART.md`](PRIOR-ART.md) | Task boundaries, evidence limits and upstream research |
| [`data/README.md`](data/README.md) · [`NOTICE.md`](NOTICE.md) | Public-data provenance and attribution |
| [`docs/render_followup_visuals.py`](docs/render_followup_visuals.py) | Rebuilds the transparent day/night result plots from saved summary CSVs |

<a id="task-boundaries"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/task-boundaries-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/task-boundaries-light-v2.svg">
  <img src="docs/assets/headings/task-boundaries-light-v2.svg" alt="Task boundaries" width="900">
</picture></h3>

- Input is processed, recorded CSI from the documented acquisition system.
- Output is one of seven predefined activity labels and its probability vector.
- Ordinary laptop RSSI does not provide this input. Body images, temperature, vital signs and unrestricted through-wall pose reconstruction are outside the evaluated task.
- Synthetic missing positions and offline repair do not establish causal real-time operation. See the [model card](MODEL_CARD.md) for full limitations.

<a id="citation"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/citation-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/citation-light-v2.svg">
  <img src="docs/assets/headings/citation-light-v2.svg" alt="Citation" width="900">
</picture></h3>

Use **Cite this repository** on GitHub or [`CITATION.cff`](CITATION.cff).

```bibtex
@software{bourkibate2026mira,
  author  = {Bourkibate, Abdessamad},
  title   = {{MIRA-CSI}: Missingness-Informed Recognition of Activities from Wi-Fi CSI},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.3.0}
}
```

Also cite **Alsaify et al.** for the [dataset](https://doi.org/10.17632/v38wjmz6f6.2) and **Dempster, Schmidt and Webb** for [MiniRocket](https://doi.org/10.1145/3447548.3467231) when using those components. Spectral covariance/temporal precedents include [PADS](https://doi.org/10.1109/PADSW.2014.7097784), [SFA](https://doi.org/10.1162/089976602317318938) and [SOBI](https://doi.org/10.1109/78.554307); [attribution](spectral/NOTICE.md) distinguishes the implementation from known ideas. Neither this software version nor the unreviewed manuscript has an assigned DOI.

<a id="license-and-contributions"></a>
<h3><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/license-and-contributions-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/license-and-contributions-light-v2.svg">
  <img src="docs/assets/headings/license-and-contributions-light-v2.svg" alt="License and contributions" width="900">
</picture></h3>

Original software: **GPL-3.0-or-later**, credited to Abdessamad Bourkibate. MiniRocket retains its original authors and license; the dataset is separately **CC BY 4.0**. See [NOTICE.md](NOTICE.md) and [data attribution](data/README.md).

Reproducible bug reports, documented replications and focused pull requests are welcome. Include the software revision, environment, input hashes and reproduction steps. Keep credentials and private measurements out of public reports.
