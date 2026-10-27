<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cover-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/cover-light.png">
    <img src="docs/assets/cover-light.png" alt="MIRA-CSI — Missingness-Informed Wi-Fi Activity Recognition, Eng. Abdessamad Bourkibate" width="900">
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
  <a href="https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.1.0"><img alt="Release" src="https://img.shields.io/badge/release-v0.1.0-176B74?style=flat-square&logo=github"></a>
  <img alt="Completed evaluation" src="https://img.shields.io/badge/evaluation-60_fits_verified-14806A?style=flat-square">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-GPL--3.0-6C5B96?style=flat-square"></a>
  <a href="https://orcid.org/0009-0000-6186-8071"><img alt="Author ORCID" src="https://img.shields.io/badge/ORCID-author-A6CE39?style=flat-square&logo=orcid&logoColor=white"></a>
</p>

<p align="center"><a href="#quick-start">Quick start</a> · <a href="#architecture">Architecture</a> · <a href="#evaluation">Results</a> · <a href="MODEL_CARD.md">Model card</a> · <a href="#citation">Citation</a></p>

## Overview

**MIRA-CSI** — *Missingness-Informed Recognition of Activities from Wi-Fi CSI* — is a compact PyTorch research model that makes observation availability explicit when classifying incomplete CSI recordings. This repository provides the model, trained states, a standalone predictor and a completed comparison against two CNN training regimes and official multivariate MiniRocket.

**Developer and research lead:** Eng. **Abdessamad Bourkibate**  
Software & Cybersecurity Engineer | AI & EdTech — **Share'In Holding**

**Status:** experimental offline classifier, evaluated on seven activity labels. The results do not establish broad model superiority or live deployment readiness.

## Quick start

Open the [executed saved-model notebook](notebooks/quickstart.ipynb) in Colab, or use the commands below. The notebook has been run top-to-bottom on a Windows CPU and retains its actual outputs.

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

## Architecture

<p align="center"><img src="figures/mira_architecture.png" alt="18 amplitude bands plus observation mask and gap channels, temporal CNN, coverage-weighted pooling and seven activity logits" width="100%"></p>

| Component | Implementation |
|---|---|
| Signal input | 18 amplitude bands × 500 normalized packet positions |
| Missingness input | Observation mask and distance to nearest observation |
| Encoder | 1D convolutions: 32 → 64 → 128 channels, GroupNorm, GELU |
| Pooling | Temporal features weighted by local observation coverage |
| Output | Seven logits, with source-validation temperature scaling |
| Model size | 40,871 trainable parameters |

The comparison includes an ordinary amplitude CNN (40,423 parameters), the same CNN with source-only loss augmentation, and the official MiniRocket transform with **9,996 features**. Mask and time-gap representations have prior art; [`PRIOR-ART.md`](PRIOR-ART.md) documents attribution and the scope of this design.

### Measurement availability, case by case

The first three clips are abstract data-flow illustrations. Teal paths represent parallel streams and dots represent observations; spacing and motion do not encode measured amplitudes, time, radio propagation or a numerical loss rate. The final plot uses the released real CSI example.

### Complete observations

<p align="center"><img src="docs/assets/case-complete.gif" alt="Conceptual complete observation streams" width="640"></p>

Available samples enter preprocessing together. Complete input means no additional mask was imposed; it does not establish that acquisition was free from noise or packet loss.

### Random missing positions

<p align="center"><img src="docs/assets/case-random-loss.gif" alt="Conceptual scattered missing observations" width="640"></p>

Scattered gaps illustrate non-contiguous unavailable positions. In the experiment, the exact requested number of positions is selected without replacement and the same temporal mask is applied to all 18 streams.

### Contiguous missing interval

<p align="center"><img src="docs/assets/case-block-loss.gif" alt="Conceptual contiguous observation gap" width="640"></p>

A common gap illustrates a continuous unavailable interval. In the evaluated 50% case, exactly 250 of 500 normalized packet positions are hidden jointly across streams. The illustration itself does not encode this fraction.

### Repair on the actual recording

<p align="center"><img src="docs/assets/case-linear-repair.gif" alt="Measured CSI band with retained observations and observed-only linear repair" width="640"></p>

Grey is the original public recording, teal is retained input, and dashed amber is linear interpolation between observed boundaries. The grey hidden values are used only for reconstruction scoring, never for repair. These offline operators can use later observed samples. Normalized packet positions are not elapsed seconds. [Visual provenance](docs/assets/CASE-VISUAL-PROVENANCE.json).

## Evaluation

**60 fitted runs · 3 excluded-environment folds · 5 paired training/mask seeds**

| Model | Complete accuracy (%) | Block 50%, linear (%) |
|---|---:|---:|
| CNN | 24.52 ± 2.17 | 22.19 ± 2.15 |
| CNN + augmentation | 22.54 ± 1.82 | 20.76 ± 1.33 |
| **MIRA-CSI** | **23.73 ± 2.26** | **23.00 ± 1.28** |
| MiniRocket (9,996) | 53.08 ± 0.33 | 19.59 ± 0.88 |

Means ± sample SD across five paired seeds; each seed pools 1,260 held-environment predictions. Room and participant cohort change together, so this protocol cannot isolate a causal room-only effect.

<p align="center"><img src="evaluation/expanded_comparison.png" alt="Measured familiar and held-environment accuracy under complete input and random or contiguous loss" width="100%"></p>

MiniRocket is stronger on complete inputs and fails substantially under long gaps. MIRA-CSI has limited absolute recognition accuracy; these experiments do not establish a substantial practical advantage over the ordinary CNN. The **[full evaluation](evaluation/README.md)** includes per-seed metrics, repair comparisons, NLL, Brier, ECE and the retained failed first-file inference example.

All **3,048 metric rows** were independently recomputed from saved probabilities in Colab and again locally. The package retains 896,112 probability vectors, including dependent supplementary mask views of the same 1,260 recordings.

## Technology stack

| Role | Languages and libraries |
|---|---|
| Model and inference | Python · PyTorch · NumPy |
| Signal preparation | Python · NumPy · SciPy |
| Reference comparison | Numba · scikit-learn · official MiniRocket source |
| Analysis and figures | pandas · Matplotlib · Pillow |
| Research documentation | Markdown · LaTeX · CFF citation metadata |

## Reproduce

Install `requirements.txt`. Obtain the exact processed public input cache from the [release assets](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.1.0), or rebuild it from pinned raw recordings:

```sh
python -m pip install -r requirements.txt
python prepare_data.py
python check_inputs.py
python run_experiment.py --models cnn_clean cnn_augmented mira_csi
python run_experiment.py --models minirocket_9996
python verify_results.py
python summarize_results.py
```

[`protocol.json`](protocol.json) fixes the selection, participant roles, model settings and seeds. The [compute amendment](protocol_amendment_01.json) pairs training/mask seeds for the primary comparison. Extra CNN masks and a previously completed full-grid MiniRocket run remain supplementary sensitivity data.

<details>
<summary><strong>Download contents and provenance</strong></summary>

| Asset | Contents |
|---|---|
| `mira-csi-verified-results.zip` | All 60 trained states, probability arrays, metrics, histories, plots and run records |
| `mira-csi-public-inputs.zip` | Exact public input tensor, manifest, attribution and hashes |
| GitHub source archive | Source code, model card, evaluation summaries and documentation |

SHA256 digests accompany the release. The input cache does not overwrite current software source. Raw-data reconstruction downloads approximately 395 MB; the optional processed cache is smaller.

</details>

## Task boundaries

- Input is processed, recorded CSI from the documented acquisition system.
- Output is one of seven predefined activity labels and its probability vector.
- Ordinary laptop RSSI does not provide this input. Body images, temperature, vital signs and unrestricted through-wall pose reconstruction are outside the evaluated task.
- Synthetic missing positions and offline repair do not establish causal real-time operation. See the [model card](MODEL_CARD.md) for full limitations.

## Citation

Use **Cite this repository** on GitHub or [`CITATION.cff`](CITATION.cff).

```bibtex
@software{bourkibate2026mira,
  author  = {Bourkibate, Abdessamad},
  title   = {{MIRA-CSI}: Missingness-Informed Recognition of Activities from Wi-Fi CSI},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.1.0}
}
```

Also cite **Alsaify et al.** for the [dataset](https://doi.org/10.17632/v38wjmz6f6.2) and **Dempster, Schmidt and Webb** for [MiniRocket](https://doi.org/10.1145/3447548.3467231) when using those components. This software release has no assigned DOI.

## License and contributions

Original software: **GPL-3.0-or-later**, credited to Abdessamad Bourkibate. MiniRocket retains its original authors and license; the dataset is separately **CC BY 4.0**. See [NOTICE.md](NOTICE.md) and [data attribution](data/README.md).

Reproducible bug reports, documented replications and focused pull requests are welcome. Include the software revision, environment, input hashes and reproduction steps. Keep credentials and private measurements out of public reports.
