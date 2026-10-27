# MIRA-CSI model card

Developer / research lead: Abdessamad Bourkibate. Status: experimental.

## Intended task

Offline recognition of seven predefined activities from the specified public
CSI recordings. The model consumes a processed `(18, 500)` amplitude array,
not images and not the laptop's ordinary RSSI reading.

It cannot display the current user, reconstruct a body, measure body water,
diagnose a fall in a live deployment or infer temperature/vital signs.

## Computation

For an observation mask `m[t]` equal to 1 for retained samples, define
`d[t] = min_{s:m[s]=1} |t-s| / 500`. After the selected offline repair, amplitudes
are standardized using each band's source-training mean and standard
deviation. The input concatenates 18 standardized streams with `m` and `d`.

Three temporal convolutions use 32, 64 and 128 channels, kernel lengths 7, 5
and 3, stride 2, GroupNorm and GELU. Local observation coverage is computed
by adaptive average pooling of `m` to the encoded temporal length. With
`w[j] = max(coverage[j], 0.05)`, the pooled representation is
`sum_j w[j] h[j] / sum_j w[j]`. Dropout and a linear head produce seven logits.

Coverage weights express where observations exist. They are not calibrated
prediction confidence, measured signal reliability or a physical inverse model.
The model has 40,871 trainable parameters; the ordinary CNN has 40,423.

Training minimizes seven-class cross-entropy with AdamW. Checkpoint selection
and temperature scaling use source validation only. No excluded-environment
labels inform the trained state or hyperparameters. Exact settings and seeds
are fixed in `protocol.json`.

## Contribution being tested

The testable proposal is the joint use of explicit missingness inputs and
coverage-weighted pooling in a compact CSI classifier. The comparison includes
the same amplitude CNN with and without corruption augmentation. This does not
establish the novelty or isolated effect of every component: there is no
separate mask-only, distance-only or pooling-only ablation in the current plan.

Mask inputs and time-gap representations already appear in prior work such as
GRU-D. Standard CNN layers and MiniRocket belong to their respective authors.
The project's authorship does not confer ownership of those prior methods or
the underlying dataset.

## Evaluation status

The five-seed, three-fold comparison completed all 60 fits. Saved probabilities
passed independent recomputation of 3,048 metric rows in Colab and on the local
computer. MIRA-CSI held-environment accuracy was 23.73% ± 2.26 on complete inputs
and 23.00% ± 1.28 with 50% contiguous loss and linear repair. These seed SDs are
conditional on this fixed subset. MiniRocket achieved 53.08% ± 0.33 and
19.59% ± 0.88, respectively. The current results do not establish practical
superiority of MIRA-CSI over the ordinary CNN or deployment readiness.

The full evaluation, including failed first-file inference and all repair
comparisons, is in `evaluation/README.md`. Authorship, an academic name and
working code do not alone establish a first scientific invention.

## Main limits

- Room and participant cohort are confounded in this dataset.
- 1,260 selected files and seven classes do not represent every Wi-Fi activity.
- Synthetic removal at normalized packet positions does not reproduce every
  burst-loss mechanism or hardware failure.
- Offline linear/nearest repair may use later observations, so this protocol
  does not establish real-time causal operation.
- Standard deviation across five seeds measures training variation conditional
  on this fixed subset and protocol; it is not population-level certainty.
- Source-only scalar temperature need not calibrate probabilities in a new room.
