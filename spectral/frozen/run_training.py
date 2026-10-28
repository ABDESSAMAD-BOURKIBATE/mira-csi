"""Resumable source-only training; held/familiar evaluation is a separate opt-in.

--evaluate performs evaluation only and requires all 75 source checkpoints to
have been frozen together first. --smoke is a labelled one-epoch source check.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for option in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[option] = '2'
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
import socket
import time
import numpy as np
from threadpoolctl import ThreadpoolController
try:
    NUMPY_POOLS = ThreadpoolController()  # Before torch in a fresh runner process.
except FileNotFoundError:
    if os.name != 'nt':
        raise
    # A unit-test host may already have imported torch, triggering its Windows
    # Unicode DLL enumeration defect. Pure helpers remain testable; main refuses
    # training in this host and requires the documented fresh child process.
    NUMPY_POOLS = None
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from models import CandidateMIRA, parameter_count
from operators import augmentation_masks, make_mask
from pipeline import (CleanFeatureCache, DataStore, dump_csv, dump_json, dump_npz,
                      prepare_batch, sha, source_normalization)

HERE = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


class StudyLock:
    """Kernel lock, released by the OS even if the owning process is killed."""
    def __init__(self, directory):
        self.path = Path(directory) / 'writer.lock'
    def __enter__(self):
        self.stream = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.stream.write(b'\0')
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError('Another writer is active in this output directory')
        self.stream.seek(1)
        self.stream.write(json.dumps(dict(pid=os.getpid(), host=socket.gethostname(), started=utc())).encode())
        self.stream.truncate()
        self.stream.flush()
        return self
    def __exit__(self, *unused):
        self.stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def atomic_torch_save(value, path):
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def distribution(logits, temperature=1.):
    z = np.asarray(logits, dtype=np.float64) / temperature
    logp = z - logsumexp(z, axis=1, keepdims=True)
    return np.exp(logp), logp


def scores(truth, probabilities, log_probabilities):
    truth = np.asarray(truth, dtype=np.int64)
    predicted = probabilities.argmax(1)
    confusion = np.zeros((7, 7), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * confusion.diagonal(), denominator,
                   out=np.zeros(7, dtype=np.float64), where=denominator > 0)
    return dict(n=len(truth), correct=int((predicted == truth).sum()),
                accuracy=float((predicted == truth).mean()), macro_f1=float(f1.mean()),
                nll=float(-log_probabilities[np.arange(len(truth)), truth].mean()),
                brier=float(((probabilities - np.eye(7)[truth])**2).sum(1).mean()))


def model_hash(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def expected_tags(config):
    return [f'{v}_E{f}_s{s}' for f in config['folds']
            for s in config['training_seeds'] for v in config['variants']]


def verify_protocol(config):
    for name, expected in config['implementation_sha256'].items():
        assert sha(HERE / name) == expected, f'Implementation differs from frozen protocol: {name}'
    assert len(expected_tags(config)) == 75
    assert config['training']['max_epochs'] == 60
    assert len(config['evaluation_conditions']) == 11


def tensors(prepared, device):
    return tuple(None if a is None else torch.as_tensor(a, device=device) for a in prepared)


@torch.inference_mode()
def infer(model, store, indices, fold, phase, mean, scale, clean_cache, device,
          condition=None, authorized=False):
    model.eval()
    outputs = []
    for begin in range(0, len(indices), 128):
        batch = np.asarray(indices[begin:begin + 128], dtype=np.int64)
        raw = store.read(batch, fold, phase, authorized=authorized)
        masks = np.ones((len(batch), 500), dtype=bool)
        if condition is not None and condition['missing_fraction']:
            masks = np.asarray([make_mask(store.entries[i]['name'], condition['pattern'],
                                          condition['missing_fraction'], condition['mask_seed'])
                                for i in batch])
        prepared = prepare_batch(raw, masks, batch, model.variant, mean, scale, clean_cache)
        outputs.append(model(*tensors(prepared, device)).detach().cpu().numpy())
    return np.concatenate(outputs)


def completed_receipt(out, tag, protocol_hash, allow_smoke=False):
    receipt = out / 'runs' / f'{tag}.training.json'
    if not receipt.exists():
        return None
    metadata = json.loads(receipt.read_text(encoding='utf-8'))
    assert metadata['protocol_sha256'] == protocol_hash
    assert metadata['mode'] == ('SMOKE_NOT_RESEARCH' if allow_smoke else 'completed_source_training')
    assert metadata['checkpoint_sha256'] == sha(out / 'checkpoints' / f'{tag}.pt')
    assert metadata['validation_sha256'] == sha(out / 'validation' / f'{tag}.npz')
    return metadata


def train_one(store, train, validation, mean, scale, variant, seed, fold,
              device, out, clean_cache, config, protocol_hash, smoke):
    tag = f'{variant}_E{fold}_s{seed}'
    finished = completed_receipt(out, tag, protocol_hash, allow_smoke=smoke)
    if finished is not None:
        print(json.dumps(dict(stage='source_resume_verified', tag=tag)), flush=True)
        return finished
    hp = config['training']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = CandidateMIRA(variant).to(device)
    initial_hash = model_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['learning_rate'],
                                 weight_decay=hp['weight_decay'])
    latest = out / 'checkpoints' / f'{tag}.latest.pt'
    final = out / 'checkpoints' / f'{tag}.pt'
    first_epoch, stale, best, best_state, best_epoch = 0, 0, float('inf'), None, None
    history = []
    if latest.exists():
        previous = torch.load(latest, map_location='cpu', weights_only=False)
        assert previous['protocol_sha256'] == protocol_hash and previous['smoke'] == smoke
        assert previous['initial_state_sha256'] == initial_hash
        np.testing.assert_array_equal(previous['mean'], mean)
        np.testing.assert_array_equal(previous['scale'], scale)
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        rng.bit_generator.state = previous['augmentation_rng']
        torch.set_rng_state(previous['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(previous['cuda_rng'])
        first_epoch, stale = previous['next_epoch'], previous['stale']
        best, best_state = previous['best'], previous['best_state']
        best_epoch, history = previous['best_epoch'], previous['history']
    epoch_limit = 1 if smoke else hp['max_epochs']
    start = time.perf_counter()
    if stale >= hp['patience']:
        first_epoch = epoch_limit
    for epoch in range(first_epoch, epoch_limit):
        model.train()
        order = rng.permutation(len(train))
        epoch_hash = hashlib.sha256()
        total_loss, total, preparation_seconds = 0., 0, 0.
        epoch_start = time.perf_counter()
        for begin in range(0, len(order), hp['batch_size']):
            indices = train[order[begin:begin + hp['batch_size']]]
            raw = store.read(indices, fold, 'train')
            masks = augmentation_masks(indices, store.entries, rng)
            epoch_hash.update(np.asarray(indices, dtype='<i8').tobytes())
            epoch_hash.update(np.packbits(masks, axis=1).tobytes())
            preparation_start = time.perf_counter()
            prepared = prepare_batch(raw, masks, indices, variant, mean, scale, clean_cache)
            preparation_seconds += time.perf_counter() - preparation_start
            target = torch.as_tensor(store.labels(indices), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(*tensors(prepared, device)), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source-training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total += len(indices)
        logits = infer(model, store, validation, fold, 'validation', mean, scale, clean_cache, device)
        p, logp = distribution(logits)
        metrics = scores(store.labels(validation), p, logp)
        record = dict(epoch=epoch + 1, train_nll=total_loss / total,
                      source_validation_nll=metrics['nll'], source_validation_accuracy=metrics['accuracy'],
                      source_validation_macro_f1=metrics['macro_f1'],
                      epoch_seconds=time.perf_counter() - epoch_start,
                      feature_preparation_seconds=preparation_seconds,
                      augmentation_epoch_sha256=epoch_hash.hexdigest())
        history.append(record)
        if metrics['nll'] < best - hp['min_delta']:
            best, stale, best_epoch = metrics['nll'], 0, epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        atomic_torch_save(dict(protocol_sha256=protocol_hash, smoke=smoke,
                               initial_state_sha256=initial_hash, mean=mean, scale=scale,
                               model=model.state_dict(), optimizer=optimizer.state_dict(),
                               augmentation_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                               next_epoch=epoch + 1, stale=stale, best=best,
                               best_state=best_state, best_epoch=best_epoch, history=history), latest)
        dump_csv(out / 'histories' / f'{tag}.csv', history)
        print(json.dumps(dict(stage='SMOKE_NOT_RESEARCH' if smoke else 'source_training',
                              tag=tag, **record)), flush=True)
        if stale >= hp['patience']:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    logits = infer(model, store, validation, fold, 'validation', mean, scale, clean_cache, device)
    truth = store.labels(validation)
    optimum = minimize_scalar(lambda t: -distribution(logits, t)[1][np.arange(len(truth)), truth].mean(),
                              bounds=(.1, 10.), method='bounded')
    if not optimum.success:
        raise RuntimeError('Source-only temperature calibration failed')
    temperature = float(optimum.x)
    probabilities, log_probabilities = distribution(logits, temperature)
    calibrated = scores(truth, probabilities, log_probabilities)
    atomic_torch_save(dict(state_dict=best_state, mean=mean, scale=scale,
                           variant=variant, training_seed=seed, held_environment=fold,
                           classes=config['classes'], temperature=temperature, best_epoch=best_epoch,
                           train_indices=train, validation_indices=validation,
                           initial_state_sha256=initial_hash, protocol_sha256=protocol_hash,
                           smoke=smoke), final)
    validation_path = out / 'validation' / f'{tag}.npz'
    dump_npz(validation_path, indices=validation, truth=truth, logits=logits,
             probabilities=probabilities, log_probabilities=log_probabilities,
             temperature=np.asarray(temperature))
    metadata = dict(mode='SMOKE_NOT_RESEARCH' if smoke else 'completed_source_training',
                    variant=variant, training_seed=seed, held_environment=fold,
                    epochs=len(history), best_epoch=best_epoch, trainable_parameters=parameter_count(model),
                    initial_state_sha256=initial_hash, temperature=temperature,
                    source_validation_nll=calibrated['nll'], source_validation_accuracy=calibrated['accuracy'],
                    source_validation_macro_f1=calibrated['macro_f1'],
                    current_process_training_seconds=time.perf_counter() - start,
                    epoch_seconds_total=sum(r['epoch_seconds'] for r in history),
                    feature_preparation_seconds_total=sum(r['feature_preparation_seconds'] for r in history),
                    source_train_files=len(train), source_validation_files=len(validation),
                    completed_utc=utc(), protocol_sha256=protocol_hash,
                    checkpoint_sha256=sha(final), validation_sha256=sha(validation_path),
                    target_or_familiar_scored=False)
    dump_json(out / 'runs' / f'{tag}.training.json', metadata)
    print(json.dumps(dict(stage='source_fit_completed', tag=tag, mode=metadata['mode'],
                          best_epoch=best_epoch, source_validation_nll=calibrated['nll'])), flush=True)
    return metadata


def verify_freeze(out, config, protocol_hash):
    path = out / 'source_fit_freeze.json'
    if not path.exists():
        raise PermissionError('Evaluation rejected: all 75 source fits must first be frozen together')
    freeze = json.loads(path.read_text(encoding='utf-8'))
    assert freeze['protocol_sha256'] == protocol_hash and freeze['completed_source_fits'] == 75
    assert set(freeze['fits']) == set(expected_tags(config))
    for tag, expected in freeze['fits'].items():
        metadata = completed_receipt(out, tag, protocol_hash)
        assert metadata is not None
        assert expected['checkpoint_sha256'] == metadata['checkpoint_sha256']
        assert expected['validation_sha256'] == metadata['validation_sha256']
        assert expected['receipt_sha256'] == sha(out / 'runs' / f'{tag}.training.json')
    return freeze


def maybe_freeze(out, config, protocol_hash):
    fits = {}
    for tag in expected_tags(config):
        metadata = completed_receipt(out, tag, protocol_hash)
        if metadata is None:
            return None
        fits[tag] = dict(checkpoint_sha256=metadata['checkpoint_sha256'],
                         validation_sha256=metadata['validation_sha256'],
                         receipt_sha256=sha(out / 'runs' / f'{tag}.training.json'),
                         temperature=metadata['temperature'], completed_utc=metadata['completed_utc'])
    path = out / 'source_fit_freeze.json'
    if path.exists():
        return verify_freeze(out, config, protocol_hash)
    assert not list((out / 'runs').glob('*.evaluation.json'))
    assert not list((out / 'predictions').glob('*.npz'))
    freeze = dict(frozen_utc=utc(), completed_source_fits=75, protocol_sha256=protocol_hash,
                  implementation_sha256=config['implementation_sha256'], input_sha256=config['input_sha256'],
                  fits=fits, held_or_familiar_scored_before_freeze=False,
                  research_status='Exploratory follow-up; historical target results were previously known')
    dump_json(path, freeze)
    return freeze


def evaluate_one(store, config, protocol_hash, out, tag, device, clean_cache, freeze_hash):
    done = out / 'runs' / f'{tag}.evaluation.json'
    prediction_path = out / 'predictions' / f'{tag}.npz'
    metric_path = out / 'metrics' / f'{tag}.csv'
    if done.exists():
        receipt = json.loads(done.read_text())
        assert receipt['source_fit_freeze_sha256'] == freeze_hash
        assert receipt['prediction_sha256'] == sha(prediction_path)
        assert receipt['metrics_sha256'] == sha(metric_path)
        print(json.dumps(dict(stage='evaluation_resume_verified', tag=tag)), flush=True)
        return
    if prediction_path.exists() or metric_path.exists():
        raise RuntimeError(f'Incomplete evaluation artifacts for {tag}; inspect before repeating inference')
    state = torch.load(out / 'checkpoints' / f'{tag}.pt', map_location='cpu', weights_only=False)
    assert state['protocol_sha256'] == protocol_hash and not state['smoke']
    fold, variant = state['held_environment'], state['variant']
    model = CandidateMIRA(variant).to(device)
    model.load_state_dict(state['state_dict'])
    held, familiar = store.test_indices(fold, authorized=True)
    indices = np.r_[held, familiar]
    held_flags = np.arange(len(indices)) < len(held)
    truths = store.labels(indices)
    probabilities_all, log_probabilities_all, logits_all, records = [], [], [], []
    start = time.perf_counter()
    for condition in config['evaluation_conditions']:
        held_logits = infer(model, store, held, fold, 'held', state['mean'], state['scale'],
                            clean_cache, device, condition, authorized=True)
        familiar_logits = infer(model, store, familiar, fold, 'familiar', state['mean'], state['scale'],
                                clean_cache, device, condition, authorized=True)
        logits = np.r_[held_logits, familiar_logits]
        p, logp = distribution(logits, state['temperature'])
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-12)
        assert np.isfinite(p).all() and np.isfinite(logp).all()
        probabilities_all.append(p)
        log_probabilities_all.append(logp)
        logits_all.append(logits)
        for domain, keep in [('held', held_flags), ('familiar', ~held_flags)]:
            records.append(dict(variant=variant, training_seed=state['training_seed'],
                                held_environment=fold, domain=domain, **condition,
                                **scores(truths[keep], p[keep], logp[keep])))
    dump_npz(prediction_path, indices=indices, truth=truths, held=held_flags,
             probabilities=np.asarray(probabilities_all), log_probabilities=np.asarray(log_probabilities_all),
             logits=np.asarray(logits_all), condition_ids=np.arange(11),
             temperature=np.asarray(state['temperature']))
    dump_csv(metric_path, records)
    dump_json(done, dict(mode='frozen_evaluation_completed', completed_utc=utc(),
                        evaluation_seconds=time.perf_counter() - start, protocol_sha256=protocol_hash,
                        source_fit_freeze_sha256=freeze_hash, checkpoint_sha256=sha(out / 'checkpoints' / f'{tag}.pt'),
                        prediction_sha256=sha(prediction_path), metrics_sha256=sha(metric_path),
                        conditions=11, imputer_duplicates=0, target_or_familiar_scored=True))
    print(json.dumps(dict(stage='evaluation_completed', tag=tag, conditions=11)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=HERE / 'data')
    parser.add_argument('--output-root', type=Path, default=HERE / 'results')
    parser.add_argument('--variants', nargs='+', choices=list('ABCDE'), default=list('ABCDE'))
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(20261026, 20261031)))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--evaluate', action='store_true', help='Evaluation-only; all 75 source fits must already be frozen')
    args = parser.parse_args()
    if NUMPY_POOLS is None:
        raise RuntimeError('Launch run_training.py in a fresh child process before importing torch')
    if args.smoke and args.evaluate:
        parser.error('Smoke mode cannot evaluate held/familiar data')
    config = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    protocol_hash = sha(HERE / 'protocol.json')
    assert protocol_hash == (HERE / 'PROTOCOL.sha256').read_text().strip(), 'Protocol receipt mismatch'
    verify_protocol(config)
    assert set(args.folds) <= set(config['folds']) and set(args.seeds) <= set(config['training_seeds'])
    out = args.output_root / (f'smoke-{protocol_hash[:12]}' if args.smoke else 'study')
    for name in ('checkpoints', 'runs', 'histories', 'validation', 'predictions', 'metrics', 'runtime'):
        (out / name).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    with StudyLock(out), NUMPY_POOLS.limit(limits=2):
        if args.evaluate:
            verify_freeze(out, config, protocol_hash)  # Gate BEFORE any test array is gathered.
        store = DataStore(args.data_root, config)
        clean_cache = CleanFeatureCache()
        runtime = dict(utc=utc(), python_version=os.sys.version, torch=torch.__version__, numpy=np.__version__,
                       device=str(device), device_name=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
                       protocol_sha256=protocol_hash, smoke=args.smoke, evaluation_only=args.evaluate,
                       torch_threads=torch.get_num_threads(), blas_threads=2)
        runtime_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'
        dump_json(out / 'runtime' / runtime_name, runtime)
        dump_json(out / 'protocol-used.json', config)
        try:
            if args.evaluate:
                freeze_hash = sha(out / 'source_fit_freeze.json')
                if not (out / 'evaluation_started.json').exists():
                    dump_json(out / 'evaluation_started.json', dict(started_utc=utc(), source_fit_freeze_sha256=freeze_hash))
                for fold in args.folds:
                    for seed in args.seeds:
                        for variant in args.variants:
                            evaluate_one(store, config, protocol_hash, out, f'{variant}_E{fold}_s{seed}',
                                         device, clean_cache, freeze_hash)
            else:
                for fold in args.folds:
                    train, validation = store.source_indices(fold)
                    mean, scale = source_normalization(store, train, fold)
                    for seed in args.seeds:
                        for variant in args.variants:
                            train_one(store, train, validation, mean, scale, variant, seed, fold,
                                      device, out, clean_cache, config, protocol_hash, args.smoke)
                            dump_json(out / 'source_data_access.json', store.audit())
                if not args.smoke:
                    frozen = maybe_freeze(out, config, protocol_hash)
                    print(json.dumps(dict(stage='source_stage_complete', all_75_frozen=frozen is not None,
                                          target_or_familiar_scored=False)), flush=True)
        finally:
            dump_json(out / ('evaluation_data_access.json' if args.evaluate else 'source_data_access.json'), store.audit())
            dump_json(out / 'clean_cache_summary.json', dict(records=len(clean_cache.items),
                                                            hits=clean_cache.hits, misses=clean_cache.misses,
                                                            augmented_features_cached=False))
    if args.smoke:
        print('SMOKE ONLY: one source epoch; no research result and no held/familiar inference.', flush=True)


if __name__ == '__main__':
    main()
