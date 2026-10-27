# MIRA-CSI

**Missingness-Informed Recognition of Activities from Wi-Fi CSI**

Developer and research lead: **Eng. Abdessamad Bourkibate**  
Software & Cybersecurity Engineer | AI & EdTech, Share'In Holding  
ORCID: https://orcid.org/0009-0000-6186-8071

MIRA-CSI is an executable research model for seven-class activity recognition
from recorded Wi-Fi channel-state information. The five-seed, three-fold
comparison is complete and independently verified. It remains experimental;
the results do not establish general superiority or deployment readiness.

| Model | Complete accuracy (%) | Block 50%, linear (%) |
|---|---:|---:|
| CNN | 24.52 ± 2.17 | 22.19 ± 2.15 |
| CNN + augmentation | 22.54 ± 1.82 | 20.76 ± 1.33 |
| MIRA-CSI | 23.73 ± 2.26 | 23.00 ± 1.28 |
| MiniRocket (9,996) | 53.08 ± 0.33 | 19.59 ± 0.88 |

Means ± SD over five paired training/mask seeds on the fixed held-environment
selection. See [the full evaluation](evaluation/README.md) for exact scope,
per-seed values, probability quality, repair errors and the retained failed
inference example. The code and model states are real; no result was imposed.

## Implemented design

A compact temporal CNN receives 18 amplitude-band streams, an observation mask
and the normalized distance to the nearest observed sample. Temporal pooling
weights encoded features by local observation coverage. This weight is a
deterministic coverage proxy; it is not a measured probability or RF field.

Two ablations distinguish the effects of source-only corruption augmentation
and the extra missingness inputs/pooling. A wider official MiniRocket reference
provides an independent algorithmic comparison. All settings are fixed in
`protocol.json` before the new results are examined.

The CNN blocks and mask-aware learning have prior art. In particular, GRU-D
already uses masking and time intervals. Our specific CSI implementation and
evaluation must be assessed against that literature; attribution is preserved
in `PRIOR-ART.md` and the vendored MiniRocket files.

## Real data and boundaries

The inputs derive from 1260 publicly released recordings by Alsaify et al.
(dataset DOI: https://doi.org/10.17632/v38wjmz6f6.2;
article DOI: https://doi.org/10.1016/j.dib.2020.106534).
The immutable selection/participant roles and raw-file hashes are preserved in
`data/manifest.json`. The portable tensor contains 18 scaled amplitude bands
at 500 normalized packet positions, not seconds or pixels.

Each of three folds excludes an entire environment and its participant cohort.
It cannot identify a causal room-only effect, because different environments
contain different participants. Artificially removed packet positions are not
empirical network outages. No private local Wi-Fi measurements are included.

## Run

Use Python with NumPy, SciPy, scikit-learn, PyTorch, Numba, joblib and Matplotlib.
In Colab these are checked in the notebook before starting. From this directory:

```sh
python prepare_data.py
python check_inputs.py
python run_experiment.py --models cnn_clean cnn_augmented mira_csi
python run_experiment.py --models minirocket_9996
python verify_results.py
python summarize_results.py
```

The protocol requires 5 seeds × 3 folds for every model (60 fitted runs).
The compute amendment in `protocol_amendment_01.json` pairs training/mask seeds
for the primary comparison: seven matched conditions per fit. CNNs retain their
additional 31-condition grid as supplementary sensitivity data; MiniRocket runs
use the seven matched conditions. Checkpoints, validation histories and complete
per-file probability arrays are saved, so aggregate claims can be independently
recomputed. Partial runs must be labeled incomplete.

`prepare_data.py` downloads only the pinned public selection, checks every raw
SHA256 and reconstructs the exact tensor recorded in `input_receipt.json`.
It requires approximately 395 MB of download/storage for the selected MAT files.

## License and citation

Original implementation is released under GPL-3.0-or-later, with author credit
to Abdessamad Bourkibate. The official MiniRocket reference retains its authors
and GPL license. Dataset ownership remains with its original authors; the
dataset is separately licensed CC BY 4.0. A citation file describes this software
without inventing a publication, DOI or release history.

## Run a saved model without retraining

Download the full verified results from the release assets and extract it in
this project directory, producing `results/`. Then run:

```sh
python predict.py --checkpoint results/checkpoints/mira_csi_E3_s20261026.pt --input results/demo_input.npy --calibration results/runs/mira_csi_E3_s20261026.json
```

Add `--observed results/demo_observed.npy` to reproduce the fixed block-loss
example. The example is a recorded public CSI segment, not the current user.
See [the executed Colab experiment](https://colab.research.google.com/drive/1HKb7rLYoDC1B8xlrZ-pyvIQqe0R9rD9w)
and the evaluation directory for the run record.
