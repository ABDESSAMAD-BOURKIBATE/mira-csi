"""Deterministic, class-independent loss masks and offline repairs."""
import hashlib
import numpy as np


def make_mask(name, pattern, fraction, seed, length=500):
    observed = np.ones(length, dtype=bool)
    if fraction == 0:
        return observed
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f'{seed}:{name}:{pattern}:{fraction}'.encode()).digest()[:8], 'little'))
    count = int(round(length*fraction))
    positions = (rng.choice(length, count, replace=False) if pattern == 'random'
                 else np.arange(start := int(rng.integers(0, length+1-count)),
                                start+count))
    observed[positions] = False
    assert (~observed).sum() == count and observed.any()
    return observed


def repair(x, observed, method, source_mean):
    result = x.copy()
    if observed.all():
        return result
    t = np.arange(x.shape[-1])
    known = t[observed]
    if method == 'linear':
        result = np.asarray([np.interp(t, known, row[observed]) for row in x],
                            dtype=np.float32)
    elif method == 'nearest':
        r = np.searchsorted(known, t).clip(0, len(known)-1)
        l = (r-1).clip(0, len(known)-1)
        choose = np.where(abs(t-known[l]) <= abs(t-known[r]), l, r)
        result[:, ~observed] = x[:, known[choose[~observed]]]
    elif method == 'source_mean':
        result[:, ~observed] = np.asarray(source_mean).reshape(-1, 1)
    else:
        raise ValueError(method)
    assert np.array_equal(result[:, observed], x[:, observed])
    assert np.isfinite(result).all()
    return result


def gap_distance(observed):
    t = np.arange(len(observed))
    known = t[observed]
    r = np.searchsorted(known, t).clip(0, len(known)-1)
    l = (r-1).clip(0, len(known)-1)
    return (np.minimum(abs(t-known[l]), abs(t-known[r])) / len(t)).astype(np.float32)
