"""Portable source-gated data access and observation-only feature preparation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
from features import extract_features
from operators import gap_distance


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def dump_csv(path, records):
    path = Path(path)
    if not records:
        return
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def dump_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


class DataStore:
    """Memmap input with explicit per-fold phase access, no pooled target stats."""
    def __init__(self, root, config):
        root = Path(root)
        for name, expected in config['input_sha256'].items():
            assert sha(root / name) == expected, f'Input hash mismatch: {name}'
        self.entries = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))['selected']
        self.values = np.load(root / 'power_band_streams.npy', mmap_mode='r')
        assert self.values.shape == (1260, 18, 500) and self.values.dtype == np.float32
        self.classes = config['classes']
        self.access = {}

    def source_indices(self, fold):
        train = np.array([i for i, e in enumerate(self.entries)
                          if e['environment'] != fold and e['role'] == 'train'], dtype=np.int64)
        validation = np.array([i for i, e in enumerate(self.entries)
                               if e['environment'] != fold and e['role'] == 'validation'], dtype=np.int64)
        assert len(train) == 504 and len(validation) == 168
        train_subjects = {(self.entries[i]['environment'], self.entries[i]['subject']) for i in train}
        val_subjects = {(self.entries[i]['environment'], self.entries[i]['subject']) for i in validation}
        assert train_subjects.isdisjoint(val_subjects)
        return train, validation

    def test_indices(self, fold, authorized=False):
        if not authorized:
            raise PermissionError('Held/familiar access requires a verified all-fit freeze')
        held = np.array([i for i, e in enumerate(self.entries) if e['environment'] == fold], dtype=np.int64)
        familiar = np.array([i for i, e in enumerate(self.entries)
                             if e['environment'] != fold and e['role'] == 'familiar_test'], dtype=np.int64)
        assert len(held) == 420 and len(familiar) == 168
        return held, familiar

    def read(self, indices, fold, phase, authorized=False):
        if phase not in ('train', 'validation', 'held', 'familiar'):
            raise ValueError(phase)
        for i in indices:
            entry = self.entries[int(i)]
            if phase in ('train', 'validation'):
                assert entry['environment'] != fold and entry['role'] == phase, 'Non-source access rejected'
            elif not authorized:
                raise PermissionError('Test access requires explicit authorization and source freeze')
            elif phase == 'held':
                assert entry['environment'] == fold
            else:
                assert entry['environment'] != fold and entry['role'] == 'familiar_test'
        key = f'E{fold}:{phase}'
        self.access.setdefault(key, set()).update(int(i) for i in indices)
        values = np.array(self.values[indices], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite source/input values')
        return values

    def labels(self, indices):
        return np.array([self.classes.index(self.entries[int(i)]['activity']) for i in indices], dtype=np.int64)

    def audit(self):
        return {key: {'unique_records': len(indices), 'indices': sorted(indices)}
                for key, indices in sorted(self.access.items())}


def source_normalization(store, train, fold):
    values = store.read(train, fold, 'train')
    mean = values.mean((0, 2), keepdims=True, dtype=np.float64)
    scale = np.maximum(values.std((0, 2), keepdims=True, dtype=np.float64), 1e-6)
    return mean, scale


class CleanFeatureCache:
    """Recordwise fixed, unfitted features, reused only for fully observed input."""
    def __init__(self):
        self.items = {}
        self.hits = 0
        self.misses = 0

    def get(self, index, values, mask, variant):
        key = (int(index), variant)
        if mask.all() and key in self.items:
            self.hits += 1
            return self.items[key]
        result = extract_features(values, mask, variant).values
        self.misses += 1
        if mask.all():
            self.items[key] = result
        return result


def prepare_batch(raw, masks, indices, variant, mean, scale, clean_cache=None):
    """Never evaluate arithmetic or feature statistics on hidden values.

    mean/scale must already be source-training statistics. The spectral branch
    receives raw observed power values, not clean-filled or target-fitted data.
    """
    raw = np.asarray(raw)
    masks = np.asarray(masks, dtype=bool)
    if raw.shape != (len(indices), 18, 500) or masks.shape != (len(indices), 500):
        raise ValueError('Incorrect source batch shape')
    observed = masks[:, None, :]
    normalized = np.zeros(raw.shape, dtype=np.float32)
    np.subtract(raw, mean, out=normalized, where=observed, casting='unsafe')
    np.divide(normalized, scale, out=normalized, where=observed, casting='unsafe')
    distance = np.asarray([gap_distance(m) for m in masks], dtype=np.float32)[:, None, :]
    features = None
    if variant != 'D':
        feature_variant = 'A' if variant == 'E' else variant
        feature_rows = []
        for j, index in enumerate(indices):
            if clean_cache is None:
                value = extract_features(raw[j], masks[j], feature_variant).values
            else:
                value = clean_cache.get(index, raw[j], masks[j], feature_variant)
            feature_rows.append(value)
        features = np.asarray(feature_rows, dtype=np.float32)
    if not np.isfinite(normalized).all() or (features is not None and not np.isfinite(features).all()):
        raise FloatingPointError('Nonfinite observation-only batch')
    return normalized, observed.astype(np.float32), distance, features
