# Changes

## 0.2.1 — 28 October 2026

- Refuse automatic reuse of incomplete, changed or unsealed historical runs.
  New completion records bind source/configuration, exact binary inputs, run
  identity and all saved artifacts; the completion record is written last.
- Verify calibration against checkpoint bytes and metadata. A hash-anchored
  manifest preserves compatibility with the exact 60 published v0.1.0 pairs;
  unknown historical calibration requires an explicit legacy opt-in.
- Normalize only UTF-8 LF/CRLF differences for text checks. Binary input hashes
  remain exact. Reject invalid supplied masks and nonfinite model outputs.
- Add a fresh output-directory option to training, verification and summary
  commands, and record the original source-fit dependency versions separately.
- Validate 13 corruption/input regressions and the existing complete/block-loss
  predictor example. These checks do not add a scientific performance experiment
  or change the frozen protocols, model computations, recorded results or v0.2.0
  evidence assets.

## 0.2.0 — 28 October 2026

Verified evaluation extension. The released compact MIRA model and v0.1.0
saved-model quick start retain their original behavior and assets.

- Add the exact frozen six-configuration, 90-fit component/residual-control study
  under `studies/component_removal/`, with portable verification and analysis.
- Add full-grid summaries, paired seed differences and conditional participant
  bootstrap ranges, retaining inconclusive and adverse component findings.
- Document the separate source-selected MiniRocket clean-input sensitivity,
  including the held domain that did not improve. This is not MIRA's score.
- Add the complete 21-page unreviewed author manuscript, matching LaTeX source,
  file hashes and actual page previews under `paper/`.
- Organize the README around inference, architecture, verified results,
  reproduction and citation; add consistent transparent light/dark result plots.
- Correct the architecture caption so the original CNN comparison is not
  mislabeled as the later component-removal experiment.

The supplementary archive provides source-frozen checkpoints and recorded
probabilities for recalculation. Optimizer resume states are excluded.
No DOI, journal acceptance, live sensing or
general superiority is claimed by this release.

## 0.1.0 — 27 October 2026

Original executable compact model, saved-model predictor, participant-disjoint
60-fit comparison, original paired training/mask summary, data attribution and
executed quick-start notebook.
