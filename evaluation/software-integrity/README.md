# Software integrity checks

These checks concern file consistency and safe reuse of research artifacts.
They do not retrain a classifier, add participants, increase recognition accuracy
or retroactively certify historical runs under a new protocol.

## Current execution

The current root runner writes its completion marker last, after hashing the
checkpoint, probabilities, metric table and applicable training history. It
records source/configuration, input and run-identity context. A resume request
checks every requested completed run before loading training arrays or fitting.
Missing, changed, partial or unsealed artifacts are refused; use a fresh
`--output-root` for a new experiment.

UTF-8 text checks accept only LF/CRLF equivalence. This fixes fresh-checkout
compatibility without treating an arbitrary edited JSON document as the same
input. Binary arrays and checkpoint files retain exact-byte checks.

## Saved-model inference

New calibration metadata binds its temperature and run identity to checkpoint
bytes. The [legacy release manifest](../legacy-release-manifest.json) contains
60 exact checkpoint/run-JSON pairs extracted only after verifying the original
v0.1.0 archive SHA256. Recognized pairs retain a historical integrity scope;
they are not reclassified as outputs of the new runner. Unknown historical
calibration needs an explicit `--allow-legacy-calibration` opt-in and emits a
warning that it is unbound.

The predictor also rejects invalid supplied masks, incompatible checkpoint
metadata and nonfinite outputs. Checksums establish file consistency; they are
not signatures authenticating a source or a substitute for scientific review.
Load serialized model files only from a trusted source.

## Validation

From the repository root:

```sh
python -m unittest discover -s tests -v
python check_inputs.py
```

The 13 regression cases cover LF/CRLF handling, byte-exact binary inputs,
truncated/missing files, orphan outputs, changed code/configuration, changed
calibration, swapped checkpoints, incompatible class/seed/fold metadata,
legacy-pair handling, invalid masks and nonfinite mock outputs. Fixtures are
synthetic program checks and are not scientific experimental evidence.

The existing public complete-input and block-loss examples were also replayed
on CPU. Their predictions remain `pick_up` and `sit_down`, while the documented
true class is `sit`; both failures are retained. The added integrity metadata
does not turn these examples into successful predictions.

The separate [90-fit frozen study](../../studies/component_removal/README.md)
and the [v0.2.0 evidence assets](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/tag/v0.2.0)
retain their original source bytes and results.

[Verification record](VERIFICATION.json) includes the clean index-checkout
check, the existing demo outcomes and the retained frozen-source hash checks.
