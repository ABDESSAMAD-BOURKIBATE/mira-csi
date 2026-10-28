"""Controlled components and a separately declared standard residual baseline.

These architectures are research comparators, not claimed novel architectures.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ControlledMIRA(nn.Module):
    def __init__(self, variant='mira_full'):
        super().__init__()
        self.variant = variant
        widths = [20, 32, 64, 128]
        layers = []
        for i, kernel in enumerate([7, 5, 3]):
            layers += [nn.Conv1d(widths[i], widths[i+1], kernel,
                       stride=2, padding=kernel//2), nn.GroupNorm(8, widths[i+1]), nn.GELU()]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(128, 7))

    def forward(self, amplitudes, observed, distance):
        mask_input = torch.zeros_like(observed) if self.variant == 'no_mask_input' else observed
        distance_input = torch.zeros_like(distance) if self.variant == 'no_distance_input' else distance
        h = self.encoder(torch.cat([amplitudes, mask_input, distance_input], dim=1))
        if self.variant == 'uniform_pool':
            pooled = h.mean(-1)
        else:
            # no_mask_input removes only that input channel; this path still uses the mask.
            w = F.adaptive_avg_pool1d(observed, h.shape[-1]).clamp_min(.05)
            pooled = (h*w).sum(-1)/w.sum(-1)
        return self.head(pooled)


class AugmentedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [18, 32, 64, 128]; layers = []
        for i, kernel in enumerate([7, 5, 3]):
            layers += [nn.Conv1d(widths[i], widths[i+1], kernel, stride=2,
                       padding=kernel//2), nn.GroupNorm(8, widths[i+1]), nn.GELU()]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(128, 7))

    def forward(self, amplitudes, observed, distance):
        return self.head(self.encoder(amplitudes).mean(-1))


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(cin, cout, 7, stride=2, padding=3),
            nn.GroupNorm(8, cout), nn.GELU(), nn.Conv1d(cout, cout, 5, padding=2),
            nn.GroupNorm(8, cout), nn.GELU(), nn.Conv1d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout))
        self.shortcut = nn.Conv1d(cin, cout, 1, stride=2)

    def forward(self, x):
        return F.gelu(self.body(x)+self.shortcut(x))


class StandardResidualCNN(nn.Module):
    """Unclaimed standard residual CNN; temporal pooling is fixed, not target-tuned."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(ResidualBlock(18,64), ResidualBlock(64,128),
                                     ResidualBlock(128,128))
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(128*8,7))

    def forward(self, amplitudes, observed, distance):
        h=self.encoder(amplitudes);length=h.shape[-1]
        # Same floor/ceil bins as adaptive_avg_pool1d, written as explicit means
        # to avoid its nondeterministic CUDA adaptive-pooling backward kernel.
        pooled=torch.stack([h[...,j*length//8:((j+1)*length+7)//8].mean(-1)
                            for j in range(8)],dim=-1)
        return self.head(pooled.flatten(1))


def build_model(variant):
    if variant == 'cnn_augmented': return AugmentedCNN()
    if variant in {'resnet1d_augmented','resnet1d_clean'}: return StandardResidualCNN()
    if variant not in {'mira_full','no_mask_input','no_distance_input','uniform_pool'}:
        raise ValueError(variant)
    return ControlledMIRA(variant)
