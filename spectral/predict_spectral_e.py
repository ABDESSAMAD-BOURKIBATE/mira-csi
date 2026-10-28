"""MIRA-CSI spectral variant E (experimental), developed by Abdessamad Bourkibate.

GPL-3.0. Uses exact frozen feature/model definitions with a stricter public
input boundary. Hashes bind these files to the recorded execution; they are
not signatures or proof of publisher authenticity. No training occurs here.
"""
from pathlib import Path
import argparse
import hashlib
import io
import importlib.util
import json
import os
import sys

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPECTED_PROTOCOL = 'cc407a98f343dcac0230c3fe2a733a429e8f980b6748ddfd8240e4542696fc32'
EXPECTED_FREEZE = 'c7575d16aefd0a12348f0e3997683cb1d6895641ebf5e0c8541780b86be15b5b'
MODEL_NAME = 'MIRA-CSI spectral variant E (experimental)'


class IntegrityError(ValueError):
    """A file or its metadata disagrees with the recorded execution."""


def verified_bytes(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise IntegrityError('File content differs from the recorded SHA256: ' + Path(path).name)
    return payload


def verified_json(path, expected):
    return json.loads(verified_bytes(path, expected).decode('utf-8'))


def checked_input(power18, observed=None):
    """Validate before Boolean conversion; hidden numeric fills are ignored."""
    values = np.asarray(power18)
    if values.shape != (18, 500):
        raise ValueError('Power input must have shape (18, 500).')
    if values.dtype.kind not in 'iuf':
        raise ValueError('Power input must contain real numeric values.')
    if observed is None:
        mask = np.ones(500, dtype=bool)
    else:
        supplied = np.asarray(observed)
        if supplied.shape != (500,) or supplied.dtype.kind not in 'biuf':
            raise ValueError('Mask must be a Boolean or real finite binary vector of length 500.')
        if not np.isfinite(supplied).all() or not np.isin(supplied, (0, 1)).all():
            raise ValueError('Mask entries must be finite and exactly 0 or 1, before Boolean conversion.')
        mask = supplied.astype(bool, copy=False)
    if not mask.any():
        raise ValueError('At least one observed position is required; total loss was not evaluated.')
    if not np.isfinite(values[:, mask]).all():
        raise ValueError('Every observed power value must be finite.')
    return values, mask


def _load_checkpoint(payload):
    """Restricted Torch loader, after SHA256 validation of these exact bytes.

    The frozen original checkpoints contain NumPy source-normalization/index
    arrays. Only their known reconstruction/type globals are allowlisted.
    There is no fallback to weights_only=False.
    """
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype('float64')), type(np.dtype('int64'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)


def _module_from_root(root, stem):
    """Load from the checked root, without reusing unqualified module caches."""
    path = root/'frozen'/f'{stem}.py'
    root_key = hashlib.sha256(str(root).encode('utf-8')).hexdigest()[:16]
    key = f'_mira_spectral_{stem}_{root_key}'
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise IntegrityError('Could not load a frozen module from the verified root.')
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module  # Required by dataclasses during module execution.
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(key, None)
        raise
    return module


class SpectralEPredictor:
    def __init__(self, fold=1, seed=20261026, package_root=None):
        if fold not in (1, 2, 3) or seed not in range(20261026, 20261031):
            raise ValueError('Only recorded folds 1--3 and training seeds 20261026--20261030 are available.')
        self.root = Path(package_root).resolve() if package_root is not None else HERE
        self.protocol = verified_json(self.root/'frozen/protocol.json', EXPECTED_PROTOCOL)
        freeze = verified_json(self.root/'artifacts/source_fit_freeze.json', EXPECTED_FREEZE)
        if freeze['protocol_sha256'] != EXPECTED_PROTOCOL or freeze['completed_source_fits'] != 75:
            raise IntegrityError('Source freeze does not bind the expected 75-fit protocol.')
        if freeze['implementation_sha256'] != self.protocol['implementation_sha256']:
            raise IntegrityError('Implementation bindings disagree.')
        for name, expected in self.protocol['implementation_sha256'].items():
            verified_bytes(self.root/'frozen'/name, expected)
        if (self.root/'frozen/PROTOCOL.sha256').read_text().strip() != EXPECTED_PROTOCOL:
            raise IntegrityError('Protocol text receipt differs.')
        self.tag = f'E_E{fold}_s{seed}'
        binding = freeze['fits'][self.tag]
        metadata = verified_json(self.root/'artifacts/runs'/f'{self.tag}.training.json',
                                 binding['receipt_sha256'])
        checkpoint = verified_bytes(self.root/'artifacts/checkpoints'/f'{self.tag}.pt',
                                    binding['checkpoint_sha256'])
        validation_payload = verified_bytes(self.root/'artifacts/validation'/f'{self.tag}.npz',
                                            binding['validation_sha256'])
        # Every needed external byte binding above is checked before Torch load.
        state = _load_checkpoint(checkpoint)
        if (state['variant'] != 'E' or state['held_environment'] != fold
                or state['training_seed'] != seed or state['smoke']
                or state['protocol_sha256'] != EXPECTED_PROTOCOL
                or state['classes'] != self.protocol['classes']):
            raise IntegrityError('Checkpoint variant, classes, fold, seed or protocol differ.')
        self.temperature = float(state['temperature'])
        if (not np.isfinite(self.temperature) or not .1 <= self.temperature <= 10
                or self.temperature != float(binding['temperature'])
                or self.temperature != float(metadata['temperature'])
                or metadata['checkpoint_sha256'] != binding['checkpoint_sha256']
                or metadata['validation_sha256'] != binding['validation_sha256']):
            raise IntegrityError('Calibration or training receipt disagrees with the freeze.')
        with np.load(io.BytesIO(validation_payload), allow_pickle=False) as val:
            if float(val['temperature']) != self.temperature:
                raise IntegrityError('Validation temperature differs.')
            if not np.array_equal(val['indices'], state['validation_indices']):
                raise IntegrityError('Validation indices differ from the checkpoint.')
        # Load the exact frozen definitions only after validating their hashes.
        feature_module = _module_from_root(self.root, 'features')
        model_module = _module_from_root(self.root, 'models')
        self.extract_features = feature_module.extract_features
        torch.set_num_threads(2)
        self.model = model_module.CandidateMIRA('E').cpu()
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.model.eval()
        self.classes = list(state['classes'])
        self.checkpoint_sha256 = binding['checkpoint_sha256']
        self.fold, self.seed = fold, seed

    def features(self, power18, observed=None):
        values, mask = checked_input(power18, observed)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                features = self.extract_features(values, mask, 'A').values
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
            raise ValueError('Feature calculation failed numerically; input magnitude or eigensolver is invalid.') from error
        if np.shape(features) != (360,) or not np.isfinite(features).all():
            raise ValueError('Feature calculation returned non-finite or incorrectly shaped values.')
        return features, mask

    @torch.inference_mode()
    def predict(self, power18, observed=None):
        features, mask = self.features(power18, observed)
        logits = self.model(features=torch.as_tensor(features[None], dtype=torch.float32)).numpy()[0]
        if np.shape(logits) != (7,) or not np.isfinite(logits).all():
            raise ValueError('Model returned non-finite or incorrectly shaped logits.')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                z = logits.astype(np.float64)/self.temperature
                centered = z-z.max()
                logp = centered-np.log(np.exp(centered).sum())
                probabilities = np.exp(logp)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('Probability calculation failed numerically.') from error
        if not np.isfinite(logp).all() or not np.isfinite(probabilities).all():
            raise ValueError('Probability calculation returned non-finite values.')
        if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1., rtol=0, atol=1e-12):
            raise ValueError('Probability calculation returned an invalid distribution.')
        predicted = int(np.argmax(probabilities))
        return dict(model=MODEL_NAME, developer='Abdessamad Bourkibate', variant='E',
                    held_environment=self.fold, training_seed=self.seed,
                    class_order=self.classes, predicted_index=predicted,
                    predicted_label=self.classes[predicted],
                    probabilities=probabilities.tolist(), log_probabilities=logp.tolist(),
                    logits=logits.tolist(), observed_positions=int(mask.sum()),
                    temperature=self.temperature, checkpoint_sha256=self.checkpoint_sha256,
                    protocol_sha256=EXPECTED_PROTOCOL, source_fit_freeze_sha256=EXPECTED_FREEZE,
                    status='Experimental offline classification; hash consistency is not authenticity.')


def main():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument('--input', required=True, type=Path, help='Unnormalized fixed log-mean-power18 .npy, shape18x500')
    parser.add_argument('--mask', type=Path, help='Optional Boolean/finite binary .npy, shape500; 1=observed')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--output', type=Path, help='Optional JSON output, otherwise stdout')
    args = parser.parse_args()
    try:
        values = np.load(args.input, allow_pickle=False)
        mask = None if args.mask is None else np.load(args.mask, allow_pickle=False)
        checked_input(values, mask)  # Fail malformed user input before model load.
        result = SpectralEPredictor(args.fold, args.seed).predict(values, mask)
        output = json.dumps(result, indent=2, allow_nan=False)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding='utf-8')
        else:
            print(output, end='')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, 'Prediction rejected: '+str(error)+'\n')


if __name__ == '__main__':
    main()
