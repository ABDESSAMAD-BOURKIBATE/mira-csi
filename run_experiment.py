"""Fixed five-seed CSI experiment. Run from the project directory in Colab.

No target labels are consulted by training, model selection or calibration.
Individual probability vectors and trained states are retained for audit.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, random, time
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss
from mira_model import ActivityCNN
from missingness import make_mask, repair, gap_distance

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
CONFIG = json.loads((ROOT/'protocol.json').read_text())
CLASSES = CONFIG['classes']
CONDITIONS = [('complete', 0., 0, 'none')] + [
    (pattern, .5, seed, method)
    for pattern in ['random', 'contiguous']
    for method in CONFIG['imputers'] for seed in CONFIG['mask_seeds']]


def write_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def probabilities(logits, temperature=1.):
    z = np.asarray(logits, dtype=np.float64)/temperature
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e/e.sum(1, keepdims=True)


def scores(y, p):
    pred = p.argmax(1); confidence = p.max(1); correct = pred == y
    ece = 0.
    for b in range(10):
        keep = (confidence >= b/10) & ((confidence < (b+1)/10)
                if b < 9 else (confidence <= 1))
        if keep.any():
            ece += keep.mean()*abs(confidence[keep].mean()-correct[keep].mean())
    return dict(n=len(y), correct=int(correct.sum()), accuracy=float(correct.mean()),
                macro_f1=float(f1_score(y, pred, labels=range(7), average='macro', zero_division=0)),
                nll=float(log_loss(y, p, labels=range(7))),
                brier=float(np.square(p-np.eye(7)[y]).sum(1).mean()), ece=float(ece))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_test(X, entries, indices, condition, mean):
    pattern, fraction, seed, method = condition
    values = []; masks = []; distances = []
    for idx in indices:
        observed = make_mask(entries[idx]['name'], pattern, fraction, seed)
        values.append(repair(X[idx], observed, method, mean) if fraction else X[idx])
        masks.append(observed.astype(np.float32)[None])
        distances.append(gap_distance(observed)[None])
    return np.asarray(values, np.float32), np.asarray(masks), np.asarray(distances)


@torch.inference_mode()
def infer(model, xx, mm, dd, mean, scale, device):
    model.eval(); values = []
    for start in range(0, len(xx), 128):
        x = torch.as_tensor((xx[start:start+128]-mean)/scale, device=device)
        m = torch.as_tensor(mm[start:start+128], device=device)
        d = torch.as_tensor(dd[start:start+128], device=device)
        values.append(model(x, m, d).cpu().numpy())
    return np.concatenate(values)


def train_cnn(X, y, entries, train, val, mean, scale, variant, seed, target, device):
    seed_all(seed)
    model = ActivityCNN(mask_aware=(variant == 'mira_csi')).to(device)
    hp = CONFIG['cnn']
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['learning_rate'],
                                 weight_decay=hp['weight_decay'])
    rng = np.random.default_rng(seed)
    tx = X[train]; ty = y[train]; names = [entries[i]['name'] for i in train]
    vx = X[val]; vm = np.ones((len(val), 1, 500), np.float32); vd = np.zeros_like(vm)
    best = float('inf'); stale = 0; history = []; state = None; best_epoch = None
    for epoch in range(hp['max_epochs']):
        model.train(); order = rng.permutation(len(train)); total = 0.; count = 0
        for start in range(0, len(order), hp['batch_size']):
            batch = order[start:start+hp['batch_size']]
            xx = tx[batch].copy(); mm = np.ones((len(batch), 1, 500), np.float32)
            dd = np.zeros_like(mm)
            if variant != 'cnn_clean':
                for j, idx in enumerate(batch):
                    if rng.random() < .5:
                        pattern = str(rng.choice(['random', 'contiguous']))
                        fraction = float(rng.choice([.1, .3, .5]))
                        method = str(rng.choice(CONFIG['imputers']))
                        aug_seed = int(rng.integers(0, 2**31-1))
                        observed = make_mask(names[idx], pattern, fraction, aug_seed)
                        xx[j] = repair(xx[j], observed, method, mean.ravel())
                        mm[j, 0] = observed; dd[j, 0] = gap_distance(observed)
            bx = torch.as_tensor((xx-mean)/scale, device=device)
            bm = torch.as_tensor(mm, device=device); bd = torch.as_tensor(dd, device=device)
            by = torch.as_tensor(ty[batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(bx, bm, bd), by)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total += float(loss.detach())*len(batch); count += len(batch)
        validation = infer(model, vx, vm, vd, mean, scale, device)
        vp = probabilities(validation)
        nll = float(log_loss(y[val], vp, labels=range(7)))
        history.append(dict(epoch=epoch+1, train_nll=total/count, validation_nll=nll,
                            validation_accuracy=float((vp.argmax(1) == y[val]).mean())))
        if nll < best-hp['min_delta']:
            best = nll; stale = 0; best_epoch = epoch+1
            state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= hp['patience']:
            break
    assert state is not None
    model.load_state_dict(state)
    tag = f'{variant}_E{target}_s{seed}'
    torch.save(dict(state_dict=state, mean=mean, scale=scale, variant=variant,
                    seed=seed, target_environment=target, best_epoch=best_epoch,
                    classes=CLASSES, protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()),
               OUT/'checkpoints'/f'{tag}.pt')
    write_rows(OUT/'histories'/f'{tag}.csv', history)
    return model, dict(epochs=len(history), best_epoch=best_epoch,
                       trainable_parameters=sum(p.numel() for p in model.parameters()))


def evaluate(predict, X, y, entries, indices, mean, vp_logits, val_y, name, seed, target, metadata):
    optimum = minimize_scalar(lambda t: log_loss(val_y, probabilities(vp_logits, t), labels=range(7)),
                              bounds=(.1,10), method='bounded')
    assert optimum.success
    temperature = float(optimum.x); rows = []; predictions = []
    domains = np.array([e['environment'] == target for e in entries])[indices]
    for cid, condition in enumerate(CONDITIONS):
        xx, mm, dd = prepare_test(X, entries, indices, condition, mean.ravel())
        p = probabilities(predict(xx, mm, dd), temperature)
        assert p.shape == (len(indices), 7) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-12)
        predictions.append(p)
        pattern, fraction, mask_seed, method = condition
        for domain, mask in [('held', domains), ('familiar', ~domains)]:
            rows.append(dict(model=name, training_seed=seed, held_environment=target,
                             domain=domain, condition_id=cid, pattern=pattern,
                             missing_fraction=fraction, mask_seed=mask_seed, imputer=method,
                             **scores(y[indices][mask], p[mask])))
    tag = f'{name}_E{target}_s{seed}'
    np.savez_compressed(OUT/'predictions'/f'{tag}.npz',
                        probabilities=np.asarray(predictions), indices=indices,
                        truth=y[indices], held=domains)
    write_rows(OUT/'metrics'/f'{tag}.csv', rows)
    metadata.update(temperature=temperature, validation_nll=float(optimum.fun),
                    clean_held_accuracy=rows[0]['accuracy'], clean_familiar_accuracy=rows[1]['accuracy'])
    (OUT/'runs'/f'{tag}.json').write_text(json.dumps(metadata, indent=2))
    print(f"DONE {tag}: held clean={100*rows[0]['accuracy']:.2f}%, "
          f"familiar={100*rows[1]['accuracy']:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=CONFIG['models'])
    parser.add_argument('--seeds', nargs='+', type=int, default=CONFIG['training_seeds'])
    parser.add_argument('--folds', nargs='+', type=int, default=CONFIG['folds'])
    args = parser.parse_args()
    for directory in ['checkpoints', 'histories', 'predictions', 'metrics', 'runs']:
        (OUT/directory).mkdir(parents=True, exist_ok=True)
    (OUT/'conditions.json').write_text(json.dumps(CONDITIONS, indent=2))
    receipt = json.loads((ROOT/'input_receipt.json').read_text())
    for key in ['band_streams.npy', 'manifest.json']:
        assert hashlib.sha256((ROOT/'data'/key).read_bytes()).hexdigest() == receipt['sha256'][key]
    X = np.load(ROOT/'data/band_streams.npy')
    entries = json.loads((ROOT/'data/manifest.json').read_text())['selected']
    assert X.shape == (1260,18,500) and np.isfinite(X).all()
    y = np.array([CLASSES.index(e['activity']) for e in entries]); env = np.array([e['environment'] for e in entries])
    roles = np.array([e['role'] for e in entries]); subjects = np.array([e['subject'] for e in entries])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    (OUT/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__, numpy=np.__version__,
         device=str(device), device_name=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
         protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()), indent=2))
    print('Device:', device, '| files:', len(entries), '| models:', args.models, flush=True)
    split_audit = []
    for target in args.folds:
        train = np.flatnonzero((env != target) & (roles == 'train'))
        val = np.flatnonzero((env != target) & (roles == 'validation'))
        familiar = np.flatnonzero((env != target) & (roles == 'familiar_test'))
        held = np.flatnonzero(env == target); indices = np.concatenate([familiar, held])
        assert [len(v) for v in [train,val,familiar,held]] == [504,168,168,420]
        groups = [set(subjects[v]) for v in [train,val,familiar,held]]
        assert all(not groups[a] & groups[b] for a in range(4) for b in range(a+1,4))
        split_audit.append(dict(target=target, participants=[sorted(int(s) for s in g) for g in groups],
                                counts=[504,168,168,420]))
        mean = X[train].mean((0,2), keepdims=True).astype(np.float32)
        scale = np.maximum(X[train].std((0,2), keepdims=True), 1e-6).astype(np.float32)
        for seed in args.seeds:
            for name in args.models:
                tag = f'{name}_E{target}_s{seed}'
                if (OUT/'runs'/f'{tag}.json').exists():
                    print('Resume verified completed run:', tag, flush=True)
                    continue
                start = time.perf_counter()
                print('START', tag, flush=True)
                if name == 'minirocket_9996':
                    import sys, joblib
                    from numba import njit
                    sys.path.insert(0, str(ROOT/'vendor/minirocket'))
                    import minirocket_multivariate as rocket
                    @njit
                    def compiler_seed(s):
                        np.random.seed(s)
                    np.random.seed(seed); compiler_seed(seed)
                    parameters = rocket.fit(np.ascontiguousarray(X[train]), num_features=10000)
                    assert len(parameters[-1]) == 9996
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.linear_model import RidgeClassifier
                    scaler = StandardScaler(with_mean=False)
                    z = scaler.fit_transform(rocket.transform(np.ascontiguousarray(X[train]), parameters))
                    classifier = RidgeClassifier(alpha=1.).fit(z, y[train])
                    def predict(xx, mm, dd):
                        return classifier.decision_function(scaler.transform(
                            rocket.transform(np.ascontiguousarray(xx), parameters)))
                    vp = predict(X[val], None, None)
                    joblib.dump(dict(parameters=parameters, scaler=scaler, classifier=classifier,
                                     classes=CLASSES), OUT/'checkpoints'/f'{tag}.joblib')
                    metadata = dict(actual_features=9996, implementation='official Numba multivariate reference')
                else:
                    model, metadata = train_cnn(X, y, entries, train, val, mean, scale,
                                               name, seed, target, device)
                    def predict(xx, mm, dd):
                        return infer(model, xx, mm, dd, mean, scale, device)
                    vp = predict(X[val], np.ones((len(val),1,500),np.float32),
                                 np.zeros((len(val),1,500),np.float32))
                evaluate(predict, X, y, entries, indices, mean, vp, y[val], name, seed, target, metadata)
                print('Elapsed seconds:', round(time.perf_counter()-start, 1), flush=True)
    (OUT/'split_audit.json').write_text(json.dumps(split_audit, indent=2))
    print('REQUESTED RUNS FINISHED. Run the independent verifier before interpreting.', flush=True)


if __name__ == '__main__':
    main()
