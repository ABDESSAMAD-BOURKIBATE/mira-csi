"""MIRA-CSI research implementation, developed for Abdessamad Bourkibate.

The blocks are standard PyTorch layers. Mask/time-gap inputs have prior art,
including GRU-D (Che et al., 2018); no first-in-field claim is made.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ActivityCNN(nn.Module):
    def __init__(self, mask_aware=False, num_classes=7):
        super().__init__()
        self.mask_aware = mask_aware
        widths = [20 if mask_aware else 18, 32, 64, 128]
        blocks = []
        for i, kernel in enumerate([7, 5, 3]):
            blocks += [nn.Conv1d(widths[i], widths[i+1], kernel,
                                 stride=2, padding=kernel//2),
                       nn.GroupNorm(8, widths[i+1]), nn.GELU()]
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, amplitudes, observed, distance):
        x = (torch.cat([amplitudes, observed, distance], dim=1)
             if self.mask_aware else amplitudes)
        h = self.encoder(x)
        if self.mask_aware:
            # Coverage is a deterministic signal-quality proxy, not a calibrated
            # confidence or probability that the reconstructed signal is true.
            weight = F.adaptive_avg_pool1d(observed, h.shape[-1]).clamp_min(0.05)
            pooled = (h * weight).sum(-1) / weight.sum(-1)
        else:
            pooled = h.mean(-1)
        return self.head(pooled)


class MIRA(ActivityCNN):
    def __init__(self, num_classes=7):
        super().__init__(mask_aware=True, num_classes=num_classes)
