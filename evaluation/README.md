# Completed evaluation

All 60 fits completed in Colab on 27 October 2026. An independent verifier
recomputed 3,048 metric rows from the retained per-file probabilities. The same
checks passed again on the local computer after the export SHA256 was verified.
There are 896,112 saved probability vectors, including supplementary repeated
mask views. These are not 896,112 independent recordings.

| Model | Complete accuracy (%) | Block 50%, linear (%) |
|---|---:|---:|
| CNN | 24.52 ± 2.17 | 22.19 ± 2.15 |
| CNN + augmentation | 22.54 ± 1.82 | 20.76 ± 1.33 |
| MIRA-CSI | 23.73 ± 2.26 | 23.00 ± 1.28 |
| MiniRocket (9,996) | 53.08 ± 0.33 | 19.59 ± 0.88 |

Values are means ± sample SD over five paired training/mask seeds. Each seed
pools 1,260 predictions from the three excluded-environment folds. The underlying
recordings and participant partitions are fixed. Room and cohort effects are
confounded; seed SD is not population-level confidence.

MIRA-CSI's block-loss macro F1 was 0.2168, compared with 0.1986 for the augmented
CNN (mean paired difference +0.0182, SD 0.0232; positive in five of five seeds).
The clean-trained CNN's block-loss F1 was 0.2162, so the mask-aware modification
does not establish a substantial practical advantage over that comparator.
MiniRocket was considerably stronger on complete inputs. None of these models
has demonstrated sufficient performance for an unconstrained live deployment.

Linear repair had the lowest hidden-value RMSE among the three evaluated repairs:
1.029 dB for random 50% loss and 2.326 dB for contiguous 50% loss. Nearest and
source-mean values, downstream F1, NLL, Brier and ECE are retained in the CSVs.
Reconstruction error is a synthetic-mask evaluation on the processed amplitude
scale; it does not measure human-body reconstruction.

The first E3 recording, selected by manifest order, was also tested through the
standalone predictor. Its true class was `sit`; the complete-input prediction was
`pick_up`, and the block-loss prediction was `sit_down`. This failed example is
retained in `demo_inference.json` rather than replaced by a correct example.

## Files

- `summary.csv`: means/SD over seeds for all models, domains, losses and repairs.
- `per_seed.csv`: metrics recomputed after pooling three folds per seed.
- `per_fold.csv` and `primary_metrics.csv`: fold-specific matched conditions.
- `ablation_deltas.csv`: paired per-seed component comparisons.
- `imputation_reconstruction*.csv`: hidden-value repair errors.
- `verification.json`: independent per-file probability audit receipt.
- `split_audit.json`: disjoint participant roles for every fold.
- `package_versions.json`: actual Colab package versions.
- `*.pdf` and `*.png`: plots computed from saved probabilities, not generated results.

The full verified export includes all model states, all probability arrays,
validation histories and execution logs. SHA256:
`2f50e1886a84837bf8e51fb80ec418fa06f8c46da9ba108e675f36f5bee1dfbd`.
