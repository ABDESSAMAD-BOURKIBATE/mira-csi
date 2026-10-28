"""File/provenance checks for current runs; historical releases stay immutable."""
from pathlib import Path
import hashlib, json, math, os


class IntegrityError(RuntimeError):
    pass


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def text_sha256(path):
    raw = Path(path).read_bytes()
    raw.decode('utf-8')
    return hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()


def text_hash_variants(path):
    raw = Path(path).read_bytes(); raw.decode('utf-8')
    lf = raw.replace(b'\r\n', b'\n')
    return {hashlib.sha256(x).hexdigest() for x in (raw, lf, lf.replace(b'\n', b'\r\n'))}


def verify_input_hash(path, expected, text=False):
    actual = file_sha256(path)
    if actual != expected and (not text or expected not in text_hash_variants(path)):
        raise IntegrityError(f'Input hash mismatch: {Path(path).name}')
    return text_sha256(path) if text else actual


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise IntegrityError(f'Missing or invalid JSON: {Path(path).name}') from exc
    if not isinstance(value, dict): raise IntegrityError(f'Expected object: {Path(path).name}')
    return value


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    raw = json.dumps(value, indent=2, allow_nan=False).encode('utf-8')
    with temporary.open('wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def run_context(root, verify_data=True):
    root = Path(root); receipt = read_json(root / 'input_receipt.json')
    inputs = {n: (verify_input_hash(root / 'data' / n, expected, text=n.endswith('.json'))
                  if verify_data or n.endswith('.json') else expected)
              for n, expected in receipt['sha256'].items()}
    names = ['run_experiment.py', 'integrity.py', 'mira_model.py', 'missingness.py', 'input_receipt.json',
             'vendor/minirocket/minirocket_multivariate.py']
    if (root / 'protocol_amendment_01.json').exists(): names.append('protocol_amendment_01.json')
    return {'protocol_sha256': text_sha256(root / 'protocol.json'),
            'source_sha256_lf': {n: text_sha256(root / n) for n in names},
            'input_sha256': inputs, 'text_hash_rule': 'UTF-8, CRLF normalized to LF only; binary inputs exact'}


def run_identity(model, seed, target):
    return {'model': model, 'training_seed': int(seed), 'held_environment': int(target)}


def artifact_paths(out, tag, model):
    suffix = '.joblib' if model == 'minirocket_9996' else '.pt'
    paths = [f'checkpoints/{tag}{suffix}', f'predictions/{tag}.npz', f'metrics/{tag}.csv']
    if model != 'minirocket_9996': paths.append(f'histories/{tag}.csv')
    return paths


def validate_temperature(value):
    if isinstance(value, bool): raise IntegrityError('Temperature must be a finite positive number')
    try: temperature = float(value)
    except (TypeError, ValueError) as exc: raise IntegrityError('Invalid temperature') from exc
    if not math.isfinite(temperature) or not .1 <= temperature <= 10:
        raise IntegrityError('Temperature must be finite and within the declared [0.1, 10] bounds')
    return temperature


def metadata_sha256(metadata):
    payload = {k: v for k, v in metadata.items() if k != '_integrity'}
    try: raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc: raise IntegrityError('Invalid run metadata values') from exc
    return hashlib.sha256(raw).hexdigest()


def verify_completed_run(out, tag, identity, context):
    out = Path(out); metadata = read_json(out / 'runs' / f'{tag}.json')
    seal = metadata.get('_integrity')
    if not isinstance(seal, dict) or seal.get('version') != 1:
        raise IntegrityError(f'{tag}: historical/unsealed output is preserved; use --output-root with a fresh directory. No automatic retrospective seal is allowed.')
    if seal.get('metadata_sha256') != metadata_sha256(metadata):
        raise IntegrityError(f'{tag}: run metadata/calibration changed after completion')
    if seal.get('context') != context or seal.get('identity') != identity:
        raise IntegrityError(f'{tag}: configuration, inputs, source code or run identity changed; use a fresh --output-root')
    if any(metadata.get(k) != v for k, v in identity.items()): raise IntegrityError(f'{tag}: run identity mismatch')
    validate_temperature(metadata.get('temperature'))
    required = artifact_paths(out, tag, identity['model'])
    if set(seal.get('artifacts', {})) != set(required): raise IntegrityError(f'{tag}: incomplete artifact inventory')
    for relative in required:
        p = out / relative
        if not p.is_file() or p.stat().st_size == 0 or file_sha256(p) != seal['artifacts'][relative]:
            raise IntegrityError(f'{tag}: missing, truncated or changed artifact: {relative}')
    checkpoint = required[0]
    if metadata.get('checkpoint_sha256') != seal['artifacts'][checkpoint]:
        raise IntegrityError(f'{tag}: checkpoint binding mismatch')
    return metadata


def completed_or_available(out, tag, identity, context):
    out = Path(out)
    if (out / 'runs' / f'{tag}.json').exists():
        verify_completed_run(out, tag, identity, context); return True
    candidates = artifact_paths(out, tag, identity['model']) + [f'checkpoints/{tag}.pt', f'checkpoints/{tag}.joblib', f'runs/{tag}.json.tmp']
    if any((out / p).exists() for p in candidates):
        raise IntegrityError(f'{tag}: partial artifacts exist without a completed seal; preserve them and use a fresh --output-root')
    return False


def seal_completed_run(out, tag, metadata, identity, context):
    out = Path(out); metadata = dict(metadata)
    validate_temperature(metadata.get('temperature'))
    required = artifact_paths(out, tag, identity['model']); hashes = {}
    for relative in required:
        p = out / relative
        if not p.is_file() or not p.stat().st_size: raise IntegrityError(f'Incomplete run artifact: {relative}')
        hashes[relative] = file_sha256(p)
    metadata.update(identity, protocol_sha256=context['protocol_sha256'], checkpoint_sha256=hashes[required[0]])
    metadata['_integrity'] = {'version': 1, 'identity': identity, 'context': context,
                              'metadata_sha256': metadata_sha256(metadata), 'artifacts': hashes}
    # The completion marker is written last, atomically, after all artifacts exist.
    atomic_json(out / 'runs' / f'{tag}.json', metadata)
    verify_completed_run(out, tag, identity, context)


def verify_checkpoint_metadata(state, root):
    import numpy as np
    root = Path(root); config = read_json(root / 'protocol.json')
    if not isinstance(state, dict): raise IntegrityError('Checkpoint must contain model state and metadata')
    if state.get('variant') not in {'cnn_clean', 'cnn_augmented', 'mira_csi'}:
        raise IntegrityError('Expected an original root CNN/MIRA checkpoint')
    if state.get('protocol_sha256') not in text_hash_variants(root / 'protocol.json'):
        raise IntegrityError('Checkpoint protocol does not match the current root protocol')
    if not isinstance(state.get('classes'), list) or state['classes'] != config['classes']:
        raise IntegrityError('Checkpoint class order mismatch')
    if state.get('seed') not in config['training_seeds'] or state.get('target_environment') not in config['folds']:
        raise IntegrityError('Checkpoint seed/fold is outside the declared root protocol')
    for name in ['mean', 'scale']:
        value = np.asarray(state.get(name))
        real = np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)
        if value.shape != (1, 18, 1) or not real or not np.isfinite(value).all():
            raise IntegrityError(f'Invalid checkpoint {name}')
        if name == 'scale' and not (value > 0).all(): raise IntegrityError('Checkpoint scales must be positive')
    expected = {n: text_sha256(root / n) for n in ['mira_model.py', 'missingness.py']}
    if 'model_source_sha256_lf' in state and state['model_source_sha256_lf'] != expected:
        raise IntegrityError('Checkpoint model/operator source mismatch')
    return run_identity(state['variant'], state['seed'], state['target_environment'])


def calibration_scope(metadata, calibration_path, checkpoint_path, state, root, allow_legacy=False):
    root = Path(root); identity = verify_checkpoint_metadata(state, root)
    validate_temperature(metadata.get('temperature'))
    checkpoint_hash = file_sha256(checkpoint_path)
    seal = metadata.get('_integrity')
    if isinstance(seal, dict):
        if seal.get('metadata_sha256') != metadata_sha256(metadata):
            raise IntegrityError('Calibration metadata changed after completion')
        context = run_context(root, verify_data=False)
        if seal.get('version') != 1 or seal.get('identity') != identity or seal.get('context') != context:
            raise IntegrityError('Calibration provenance/configuration does not match the current run')
        if any(metadata.get(k) != v for k, v in identity.items()): raise IntegrityError('Calibration run identity mismatch')
        tag = f"{identity['model']}_E{identity['held_environment']}_s{identity['training_seed']}"
        if metadata.get('checkpoint_sha256') != checkpoint_hash or seal.get('artifacts', {}).get(f'checkpoints/{tag}.pt') != checkpoint_hash:
            raise IntegrityError('Calibration is bound to a different checkpoint')
        return 'current_checkpoint_bound'
    # A known legacy pair is checked against the immutable, hash-anchored release,
    # without pretending that it was produced by the new sealed runner.
    registry = read_json(root / 'evaluation/legacy-release-manifest.json')
    if registry.get('archive_sha256') != '2f50e1886a84837bf8e51fb80ec418fa06f8c46da9ba108e675f36f5bee1dfbd':
        raise IntegrityError('Legacy registry archive provenance mismatch')
    tag = f"{identity['model']}_E{identity['held_environment']}_s{identity['training_seed']}"
    pair = registry.get('pairs', {}).get(tag, {})
    if pair.get('checkpoint_sha256') == checkpoint_hash and pair.get('run_json_sha256') == file_sha256(calibration_path):
        return 'exact_legacy_v0.1.0_released_pair; no_current_source_seal'
    if not allow_legacy: raise IntegrityError('Unknown historical calibration/checkpoint pair. Use a verified released pair or explicitly --allow-legacy-calibration for trusted unbound legacy metadata.')
    if Path(calibration_path).stem != tag: raise IntegrityError('Legacy calibration filename belongs to another run')
    return 'legacy_unbound_explicit_opt_in'
