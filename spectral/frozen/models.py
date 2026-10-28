"""Untrained candidate models A-E; architecture definitions, not trained models.

The temporal path reproduces the existing MIRA-CSI Conv/GroupNorm/GELU design.
Masks/gaps/coverage pooling and MLP fusion are prior art. No physical-confidence
or receptive-field-locality guarantee follows from GroupNorm across time.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [20, 32, 64, 128]
        blocks = []
        for i, kernel in enumerate((7, 5, 3)):
            blocks.extend((nn.Conv1d(widths[i], widths[i + 1], kernel,
                                     stride=2, padding=kernel // 2),
                           nn.GroupNorm(8, widths[i + 1]), nn.GELU()))
        self.encoder = nn.Sequential(*blocks)

    def forward(self, amplitudes, observed, distance):
        if amplitudes.ndim != 3 or amplitudes.shape[1] != 18:
            raise ValueError('Expected batch x 18 x time source-normalized inputs')
        if observed.shape != (len(amplitudes), 1, amplitudes.shape[-1]):
            raise ValueError('Expected a shared temporal mask')
        if distance.shape != observed.shape:
            raise ValueError('Distance shape must match mask shape')
        # Source normalization must be fitted outside this module on training data.
        # Canonical zero-fill occurs afterwards and prevents hidden values entering CNN.
        canonical = torch.where(observed.bool(), amplitudes, torch.zeros_like(amplitudes))
        encoded = self.encoder(torch.cat((canonical, observed, distance), dim=1))
        weight = F.adaptive_avg_pool1d(observed, encoded.shape[-1]).clamp_min(0.05)
        return (encoded * weight).sum(-1) / weight.sum(-1)


class CandidateMIRA(nn.Module):
    """A/B/C differ in fixed features only; D is temporal-only; E is feature-only.

    Features MUST be extracted for the actual mask of each input, after corruption.
    Supplying clean-record features for a corrupted record would leak hidden values.
    This architecture module performs no scaling fit, calibration, or data loading.
    """
    def __init__(self, variant='A', num_classes=7, feature_size=360):
        super().__init__()
        if variant not in 'ABCDE' or len(variant) != 1:
            raise ValueError(variant)
        self.variant = variant
        self.feature_size = feature_size
        if variant != 'E':
            self.temporal = TemporalEncoder()
        if variant != 'D':
            self.feature_encoder = nn.Sequential(nn.Linear(feature_size, 64), nn.GELU(),
                                                 nn.Linear(64, 32), nn.GELU())
        head_width = 32 if variant == 'E' else 128 if variant == 'D' else 160
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(head_width, num_classes))

    def forward(self, amplitudes=None, observed=None, distance=None, features=None):
        parts = []
        if self.variant != 'E':
            parts.append(self.temporal(amplitudes, observed, distance))
        if self.variant != 'D':
            if features is None or features.ndim != 2 or features.shape[1] != self.feature_size:
                raise ValueError(f'Expected batch x {self.feature_size} actual-mask features')
            parts.append(self.feature_encoder(features))
        combined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return self.head(combined)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
