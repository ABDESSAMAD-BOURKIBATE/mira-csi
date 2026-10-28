# Completed component controls — MIRA-CSI

This directory contains the exact frozen source and analysis for the completed
28 October 2026 exploratory follow-up. Developer/research lead: **Abdessamad
Bourkibate**. These files accompany the research findings; this directory is
not a new accuracy claim or a replacement name for MiniRocket.

## Study index

| Study | Scope | Relationship |
|---|---|---|
| Original v0.1.0 | 60 fits: clean/augmented compact CNNs, full MIRA and official MiniRocket | The original release and root software protocol remain distinct. |
| This follow-up | 90 fits: four controlled MIRA configurations and two conventional residual CNNs | Same 1,260 recordings, participant-disjoint source roles, three outer folds and five training seeds; not 90 new participant samples. |
| Source-selected MiniRocket representation sensitivity | A separate five-candidate source screen and 30 sealed clean comparison configurations | Included in the supplement; it is a known-transform sensitivity, not a MIRA architecture contribution. |

The completed follow-up retains 5,580 metric rows and 1,640,520 saved probability
vectors. It is post-hoc exploratory because earlier target results were already
known. The explicit mask-input removal leaves distance and coverage information
available. Omitting the distance input slightly improves block-loss macro F1
in the declared comparison; the mask-input contrast is inconclusive, and the
pooling contrast's three-comparison-adjusted range includes zero. All controls
and negative findings are retained. No general superiority, new-room guarantee
or causal room effect is established.

## Files and frozen provenance

- `models.py`, `operators.py`, `run_ablation.py` and `protocol.json` are
  **byte-for-byte** copies of the executed source-training freeze. Their
  original hashes and historical pre-execution status strings remain intact.
- `analysis_evaluation.py` is the byte-identical analysis that reproduced the
  verified saved evaluation and 10,000 participant-cluster bootstrap draws.
  It does not train or infer a model on target recordings.
- `STUDY-SOURCES.json` records the file hashes, input hashes and exact
  supplementary archive names/checksums. `check_study.py` verifies them
  without importing a model or changing results.
- `VALIDATION.json` records the relocated-code input check, analysis helper
  check and separate source-only smoke. A smoke run is not a completed fit
  and does not contribute to the reported scientific results.

The files were moved without changing path defaults. **Always pass
`--data-root` and `--output-root` explicitly** in this location; the historical
defaults were written for the original experiment layout. Similarly, pass
`--protocol` explicitly to the analysis. The runner's default variants are
only the three primary removals; reproducing all 90 fits requires all six
variant names listed below.

## Supplementary assets

The approximately 227 MB supplement is a downloadable [v0.2.0 release asset](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.2.0/MIRA-CSI-reproducibility-supplement-20261028-validated.zip), **not Git source files**:

`MIRA-CSI-reproducibility-supplement-20261028-validated.zip`

SHA256:
`b06802d9e3cd064d94c3caead3f37c4dee5bea7b7fe3f73a157dbfbf4d79ee9e`

Obtain the asset from the [v0.2.0 release](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.2.0).
Extract it under `studies/component_removal/local-assets/supplement/`.
That directory and all local runs/checkpoints/caches are ignored by this study's
`.gitignore`. The article PDF is separate; preparation of the supplement does
not assign a DOI or peer-review status.

The extracted supplement contains:

```text
archives/MIRA-CSI-source-training-20261028.zip
archives/MIRA-CSI-source-trained-final-20261028.zip
archives/MIRA-CSI-verified-evaluation-20261028.zip
analysis/participant-bootstrap-indices.npz
```

The middle archive contains 90 final source checkpoints and metadata. Exactly
90 optimizer/RNG resume checkpoints were excluded, without reserializing any
retained checkpoint. Historical receipts name the original full archive;
`SOURCE-ARCHIVE-DERIVATION.json` maps its unchanged member bytes to the public
subset. The original evaluation archive does not include post-evaluation
checkpoint bytes/hash manifests; its receipt states that verification limit.

The source-training archive stores the single processed input cache. If the
repository's `data/` already contains the exact cache, reuse it. Otherwise run
the supplement's `tools/prepare_inputs.py` and pass that supplement's
`mira-csi/data/` directory as `--data-root`; no second cache needs to be copied
into Git. All examples below run **from the repository root** and assume the
existing verified `data/` directory.

## Validate before execution

Use Python 3.12 and the analysis environment pinned in this directory's
`requirements.txt` (the same dependency list as the supplement's
`requirements-analysis.txt`). The recorded CPU verification used PyTorch 2.11.0;
source training used PyTorch 2.11.0 on a T4. No installed environment or model
weights are stored in this Git directory.

Install dependencies in a separate activated environment, from the repository
root:

```sh
python -m pip install -r studies/component_removal/requirements.txt
```

```sh
python studies/component_removal/check_study.py --data-root data
python studies/component_removal/analysis_evaluation.py --self-check
```

The analysis helper test uses explicitly synthetic arrays to check scoring
arithmetic and interval quantiles; it is not experimental evidence. To check
the actual downloaded nested archives as well:

```sh
python studies/component_removal/check_study.py --data-root data --asset-root studies/component_removal/local-assets/supplement
```

## Separate source-only smoke

```sh
python studies/component_removal/run_ablation.py --data-root data --output-root studies/component_removal/local-runs/execution-check --smoke --variants mira_full --folds 1 --seeds 20261026
```

This writes one epoch under `local-runs/execution-check/smoke/`, uses source
training/validation only, and leaves `metrics/` and `predictions/` empty. The
runner rejects `--smoke --evaluate`. The diagnostic must not be mixed with
the completed source-fit archive or counted as an additional scientific fit.

## Recalculate the completed saved results

```sh
python studies/component_removal/analysis_evaluation.py --source-archive studies/component_removal/local-assets/supplement/archives/MIRA-CSI-source-trained-final-20261028.zip --evaluation-archive studies/component_removal/local-assets/supplement/archives/MIRA-CSI-verified-evaluation-20261028.zip --data-root data --protocol studies/component_removal/protocol.json --output-root studies/component_removal/local-runs/recomputed-analysis --bootstrap-indices studies/component_removal/local-assets/supplement/analysis/participant-bootstrap-indices.npz --bootstrap 10000 --seed 20261028
```

Use a new output directory. This reads fixed predictions and checkpoints;
it performs no new fit, temperature selection or target inference. It
reproduces the declared full-grid estimand and three B50-linear macro-F1
component contrasts. Secondary intervals remain unadjusted. Its conditional
ranges do not cover new environments or model retraining.

## Fresh source training, optional

The following is the full six-configuration, three-fold, five-seed source-only
training command. It can take substantial compute and is not required to
audit the saved results:

```sh
python studies/component_removal/run_ablation.py --data-root data --output-root studies/component_removal/local-runs/source-rerun --variants mira_full no_mask_input no_distance_input uniform_pool resnet1d_clean resnet1d_augmented --folds 1 2 3 --seeds 20261026 20261027 20261028 20261029 20261030
```

No target inference occurs without the explicit `--evaluate` flag, which is
deliberately absent above. A fresh run must freeze all source-selected
weights/calibration before its evaluation. No target scores may determine
which settings or results are retained. Cross-device training is not promised
to reproduce checkpoints bit-for-bit. Do not reuse smoke outputs as fits.

## Attribution and units

Code retains the repository's [GPL-3.0-or-later license](../../LICENSE).
Data/scaling provenance remains separate: Alsaify et al., Mendeley Data v2,
CC BY 4.0, [doi:10.17632/v38wjmz6f6.2](https://doi.org/10.17632/v38wjmz6f6.2),
and [Data in Brief 33:106534](https://doi.org/10.1016/j.dib.2020.106534).
The pinned author repository commit is
`5337147c5156948b13e6b05a50b79529614acfbe`.
See [data attribution](../../data/README.md) and
[the scaling-source MIT notice](../../data/SOURCE-CODE-LICENSE.txt).

The 18x500 tensors are relative scaled-channel dB amplitude features over
normalized packet-index positions; they are not calibrated dBm, temperatures,
body-surface coordinates or true elapsed-time sequences. Artificial masking
does not constitute a measured live network outage. Body images, vital signs,
real-time deployment and unrestricted multi-person sensing are not evaluated.

Masks/gap information have precedents including GRU-D; this study tests the
implemented CSI combination, not ownership of those ideas. MiniRocket is by
Dempster, Schmidt and Webb and belongs to its original authors. No data or
third-party method ownership is transferred by this directory or its name.
