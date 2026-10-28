# Model card: MIRA-CSI spectral variant E (experimental)

Developer: Abdessamad Bourkibate. Seven-class offline classifier; 25,415
trainable parameters. Fixed 360-dimensional observed-endpoint features feed
a 360–64–32 GELU MLP, dropout 0.2 and a seven-logit head. Each checkpoint has
its own clean source-validation temperature, used unchanged under loss.

The implementation grid uses whole/half-record cells and lags 1, 2, 4, 8, 16, 32.
For observed endpoint pairs `(a_i,b_i)`, the numerator is the uncentered
increment second moment `sum((b_i-a_i)(b_i-a_i)^T)/(2n)`; the denominator is
the covariance of the same multiset of endpoints around its endpoint mean.
The full effective-support generalized spectrum is in [0,2]. Relative-rank
threshold 1e-8, log mixture 1e-3, minimum 36 pairs, neutral padding, validity flags
and pair fractions are part of the frozen implementation.

Covariance-normalized temporal statistics are known from SFA/SOBI; PADS is
earlier CSI covariance/eigen-feature work. This is an implementation and an
exploratory comparison of the specified endpoint normalization and classifier
paths. It does not claim invention of generalized eigenvalues, masks, MLPs or
CNNs, or first use of covariance features in CSI.

Both matched A and all-observed B normalization have ideal full-support affine
congruence invariance. The relative threshold can change the retained subspace
even when retained rank is unchanged. Matching provides a coherent endpoint
measure and [0,2] bound, not invariance absent from B. A–B also changes endpoint
weighting/centering on complete records. C changes ordering and rank/neutral
rules as well as off-diagonal structure. The hybrid CNN is not generally
invariant. These statistics do not model arbitrary physical room changes.

The development direction followed historically known target results. E was
among the frozen arms before its evaluation but was highlighted after its
favourable result. The primary A–B contrast was small. New independent data
and appropriate matched-input controls are needed to test reproducibility and
mechanism. Different people occupy each environment, preventing a causal
separation of room and cohort. The seven selected segments are not the full
dataset benchmark.

Loss masks are synthetic post-resampling deletions of normalized packet
positions, not observed acquisition outages. The fixed representation does
not preserve irregular packet time or recording duration. The model does not
reconstruct missing motion, estimate physical confidence, or validate
continuous live sensing. Availability flags are not calibrated correctness
probabilities or participant sample counts.

The public wrapper validates masks before conversion to Boolean. The exact
historical training preparation wrapper cast first; the actual study used
Boolean masks, and no current-run hidden-value leak was demonstrated. That
frozen code is preserved for reproducibility rather than silently patched.

This model is not evaluated for medical decisions, identity recognition,
surveillance of nonconsenting people, thermal images or dense pose. The demo
and all-record results must not be presented as a live depiction of a user.
