"""Candidate support-matched lag-covariance features, not an established advance.

Covariance quotients/eigenspectra have prior art in SFA/SOBI. This experiment
isolates support matching under synthetic loss. No physical confidence, room
invariance, recovery, causal streaming, or first-in-field claim is made.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    channels: int = 18
    length: int = 500
    lags: tuple = (1, 2, 4, 8, 16, 32)
    cells: tuple = ((0, 500), (0, 250), (250, 500))
    min_pairs: int = 36
    rank_relative_threshold: float = 1e-8
    log_mixture_epsilon: float = 1e-3
    eigen_bound_tolerance: float = 1e-4

    @property
    def feature_size(self):
        return len(self.cells) * len(self.lags) * (self.channels + 2)

    def validate(self):
        assert self.channels > 0 and self.length > 1
        assert self.min_pairs > 0
        assert 0 < self.rank_relative_threshold < 1
        assert 0 < self.log_mixture_epsilon < 1
        assert all(0 <= start < stop <= self.length for start, stop in self.cells)
        assert all(0 < lag < stop - start
                   for start, stop in self.cells for lag in self.lags)


@dataclass
class FeatureResult:
    values: np.ndarray
    diagnostics: list


def pair_indices(observed, start, stop, lag):
    """Keep original positions; unobserved interior points are not compressed."""
    left = np.arange(start, stop - lag, dtype=np.int64)
    right = left + lag
    keep = observed[left] & observed[right]
    return left[keep], right[keep]


def pair_moments(x, left, right):
    """Float64 moments with endpoint multiplicities, same pair set in both."""
    if len(left) == 0:
        raise ValueError('At least one observed pair is required')
    a = x[:, left].T.astype(np.float64, copy=False)
    b = x[:, right].T.astype(np.float64, copy=False)
    mean = (a.sum(0) + b.sum(0)) / (2 * len(left))
    difference = b - a
    a = a - mean
    b = b - mean
    numerator = difference.T @ difference / (2 * len(left))
    denominator = (a.T @ a + b.T @ b) / (2 * len(left))
    return mean, (numerator + numerator.T) / 2, (denominator + denominator.T) / 2


def effective_spectrum(numerator, denominator, config, bounded):
    """Quotient-space spectrum; numerical rank truncation limits invariance."""
    variance, basis = np.linalg.eigh((denominator + denominator.T) / 2)
    maximum = float(variance[-1])
    if maximum <= 0:
        return np.empty(0), dict(rank=0, clipped_eigenvalues=0,
                                 retained_condition=None)
    if variance[0] < -config.eigen_bound_tolerance * maximum:
        raise FloatingPointError('Denominator is not numerically positive semidefinite')
    keep = variance > config.rank_relative_threshold * maximum
    retained = variance[keep]
    whitener = basis[:, keep].T / np.sqrt(retained)[:, None]
    core = whitener @ numerator @ whitener.T
    spectrum = np.linalg.eigvalsh((core + core.T) / 2)
    if spectrum.min(initial=0.) < -config.eigen_bound_tolerance:
        raise FloatingPointError('Negative generalized eigenvalue exceeds tolerance')
    if bounded and spectrum.max(initial=0.) > 2 + config.eigen_bound_tolerance:
        raise FloatingPointError('Matched-pair spectrum exceeds its mathematical bound')
    clipped = (spectrum < 0) | ((spectrum > 2) if bounded else False)
    spectrum = np.clip(spectrum, 0, 2 if bounded else None)
    return spectrum, dict(rank=int(keep.sum()),
                          clipped_eigenvalues=int(clipped.sum()),
                          retained_condition=float(retained[-1] / retained[0]))


def extract_features(values, observed, variant='A', config=None):
    """Extract A/B/C using only observed entries of an individual recording.

    A: endpoint-matched generalized covariance spectrum.
    B: same numerator, denominator from all observed positions in the cell.
    C: diagonal per-channel ratios with the same matched endpoints (simple control).
    Output order: all spectral/ratio values, then validity flags, then pair fractions.
    Hidden NaN/Inf values are permitted: they are never used in calculations.
    """
    config = config or FeatureConfig()
    config.validate()
    if variant not in ('A', 'B', 'C'):
        raise ValueError(variant)
    x = np.asarray(values)
    raw_mask = np.asarray(observed)
    if x.shape != (config.channels, config.length):
        raise ValueError(f'Expected {(config.channels, config.length)}, got {x.shape}')
    if raw_mask.shape != (config.length,) or not np.isin(raw_mask, [0, 1]).all():
        raise ValueError('Observation mask must be one binary temporal vector')
    mask = raw_mask.astype(bool, copy=False)
    if not np.isfinite(x[:, mask]).all():
        raise ValueError('Observed entries must be finite')

    features, flags, counts, diagnostics = [], [], [], []
    for cell_number, (start, stop) in enumerate(config.cells):
        global_denominator = None
        if variant == 'B':
            known = np.arange(start, stop)[mask[start:stop]]
            if len(known):
                known_values = x[:, known].T.astype(np.float64, copy=False)
                centered = known_values - known_values.mean(0)
                global_denominator = centered.T @ centered / len(known)
        for lag in config.lags:
            left, right = pair_indices(mask, start, stop, lag)
            count = len(left)
            fraction = count / (stop - start - lag)
            item = dict(cell=cell_number, start=start, stop=stop, lag=lag,
                        pairs=count, pair_fraction=float(fraction), valid=False,
                        rank=None, clipped_eigenvalues=0, retained_condition=None)
            feature = np.zeros(config.channels, dtype=np.float64)
            if count >= config.min_pairs:
                if variant == 'C':
                    # No off-diagonal moment or matrix rank enters this control.
                    a = x[:, left].T.astype(np.float64, copy=False)
                    b = x[:, right].T.astype(np.float64, copy=False)
                    center = (a.sum(0) + b.sum(0)) / (2 * count)
                    denominator = ((a - center)**2 + (b - center)**2).sum(0) / (2 * count)
                    numerator = ((b - a)**2).sum(0) / (2 * count)
                    nonzero = denominator > 0
                    if nonzero.any():
                        spectrum = np.ones(config.channels)
                        spectrum[nonzero] = numerator[nonzero] / denominator[nonzero]
                        if (spectrum < -config.eigen_bound_tolerance).any() or (
                                spectrum > 2 + config.eigen_bound_tolerance).any():
                            raise FloatingPointError('Diagonal ratio exceeds matched bounds')
                        item['clipped_eigenvalues'] = int(((spectrum < 0) | (spectrum > 2)).sum())
                        spectrum = np.clip(spectrum, 0, 2)
                        feature = np.log((1 - config.log_mixture_epsilon) * spectrum
                                         + config.log_mixture_epsilon)
                        item.update(valid=True, active_diagonal_channels=int(nonzero.sum()))
                else:
                    _, numerator, matched_denominator = pair_moments(x, left, right)
                    denominator = matched_denominator if variant == 'A' else global_denominator
                    spectrum, detail = effective_spectrum(numerator, denominator, config,
                                                          bounded=(variant == 'A'))
                    item.update(detail)
                    if len(spectrum):
                        feature[:len(spectrum)] = np.log(
                            (1 - config.log_mixture_epsilon) * spectrum
                            + config.log_mixture_epsilon)
                        item.update(valid=True, minimum_eigenvalue=float(spectrum.min()),
                                    maximum_eigenvalue=float(spectrum.max()))
            features.extend(feature)
            flags.append(float(item['valid']))
            counts.append(fraction)
            diagnostics.append(item)
    result = np.asarray(features + flags + counts, dtype=np.float32)
    if result.shape != (config.feature_size,) or not np.isfinite(result).all():
        raise FloatingPointError('Invalid feature output')
    return FeatureResult(result, diagnostics)
