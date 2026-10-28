"""Class-independent normalized-index masks and source augmentation draws."""
import hashlib
import numpy as np


def make_mask(name, pattern, fraction, seed, length=500):
    observed = np.ones(length, dtype=bool)
    if fraction == 0:
        return observed
    if pattern not in ('random', 'contiguous') or not 0 < fraction < 1:
        raise ValueError('Unsupported corruption')
    digest = hashlib.sha256(f'{seed}:{name}:{pattern}:{fraction}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    count = int(round(length * fraction))
    if pattern == 'random':
        positions = rng.choice(length, count, replace=False)
    else:
        start = int(rng.integers(0, length + 1 - count))
        positions = np.arange(start, start + count)
    observed[positions] = False
    assert (~observed).sum() == count and observed.any()
    return observed


def gap_distance(observed):
    t = np.arange(len(observed))
    known = t[observed]
    if not len(known):
        # An explicit sentinel only; the planned fractions never remove all data.
        return np.ones(len(observed), dtype=np.float32)
    right = np.searchsorted(known, t).clip(0, len(known) - 1)
    left = (right - 1).clip(0, len(known) - 1)
    return (np.minimum(abs(t - known[left]), abs(t - known[right])) / len(t)).astype(np.float32)


def augmentation_masks(indices, entries, rng):
    """The independent NumPy generator follows exactly the same order for A-E.

    No imputer is sampled: every variant is observed-only. Torch/dropout RNG
    consumption cannot change this generator's augmentation sequence.
    """
    masks = np.ones((len(indices), 500), dtype=bool)
    for j, index in enumerate(indices):
        if rng.random() < .5:
            pattern = str(rng.choice(['random', 'contiguous']))
            fraction = float(rng.choice([.1, .3, .5]))
            mask_seed = int(rng.integers(0, 2**31 - 1))
            masks[j] = make_mask(entries[index]['name'], pattern, fraction, mask_seed)
    return masks
