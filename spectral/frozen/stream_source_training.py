"""Colab live-output source-only entry; never adds --evaluate."""
from pathlib import Path
import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pipeline import dump_json, sha


def run(root, output_root, print_every=5, download=False):
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    expected_protocol = (root / 'PROTOCOL.sha256').read_text().strip()
    assert sha(root / 'protocol.json') == expected_protocol
    config = json.loads((root / 'protocol.json').read_text())
    for name, expected in config['implementation_sha256'].items():
        assert sha(root / name) == expected, f'Implementation changed: {name}'
    tags = {f'{v}_E{f}_s{s}' for f in config['folds'] for s in config['training_seeds'] for v in config['variants']}
    assert len(tags) == 75
    study = output_root / 'study'
    verified, signatures = {}, {}

    def completed():
        for tag in tags:
            receipt = study / 'runs' / f'{tag}.training.json'
            state = study / 'checkpoints' / f'{tag}.pt'
            if not receipt.is_file() or not state.is_file():
                continue
            signature = (receipt.stat().st_mtime_ns, state.stat().st_mtime_ns,
                         receipt.stat().st_size, state.stat().st_size)
            if signatures.get(tag) == signature:
                continue
            metadata = json.loads(receipt.read_text())
            assert metadata['mode'] == 'completed_source_training'
            assert metadata['protocol_sha256'] == expected_protocol
            assert metadata['checkpoint_sha256'] == sha(state)
            signatures[tag] = signature
            verified[tag] = metadata
        return set(verified)

    output_root.mkdir(parents=True, exist_ok=True)
    seen = completed()
    print(f'Verified source fits: {len(seen)}/75. No held/familiar evaluation is requested.', flush=True)
    command = [sys.executable, '-u', str(root / 'run_training.py'), '--data-root', str(root / 'data'),
               '--output-root', str(output_root)]
    assert '--evaluate' not in command and '--smoke' not in command
    environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    messages, end = queue.Queue(), object()
    with (output_root / 'source-training-stream.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', bufsize=1,
                                 env=environment, cwd=root)

        def collect():
            try:
                for line in child.stdout:
                    messages.put(line)
            finally:
                messages.put(end)
        threading.Thread(target=collect, daemon=True).start()
        ended = False
        try:
            while not ended or child.poll() is None:
                try:
                    line = messages.get(timeout=2)
                except queue.Empty:
                    line = None
                if line is end:
                    ended = True
                elif line is not None:
                    log.write(line)
                    log.flush()
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        record = None
                    if isinstance(record, dict) and record.get('stage') == 'source_training':
                        if record['epoch'] == 1 or record['epoch'] % print_every == 0:
                            print(f"{record['tag']} | epoch {record['epoch']:02d} | "
                                  f"source NLL {record['source_validation_nll']:.4f} | "
                                  f"source accuracy {100 * record['source_validation_accuracy']:.2f}% | "
                                  f"{record['epoch_seconds']:.2f}s/epoch", flush=True)
                    else:
                        print(line.rstrip(), flush=True)
                now = completed()
                for tag in sorted(now - seen):
                    seen.add(tag)
                    print(f'COMPLETED SOURCE FIT {len(seen)}/75: {tag}', flush=True)
            return_code = child.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            print('Only this child stopped; atomically completed epoch states remain resumable.', flush=True)
            raise
    assert completed() == tags
    freeze_path = study / 'source_fit_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['completed_source_fits'] == 75 and freeze['protocol_sha256'] == expected_protocol
    dump_json(output_root / 'source-stream-completion.json',
              dict(completed_source_fits=75, protocol_sha256=expected_protocol,
                   source_fit_freeze_sha256=sha(freeze_path), held_or_familiar_evaluation_requested=False))
    print('ALL 75 SOURCE FITS FROZEN. No held/familiar result was computed by this entry.', flush=True)
    if download:
        import shutil
        archive = shutil.make_archive(str(output_root.parent / 'MIRA-v2-source-trained-75'), 'zip',
                                      root_dir=output_root)
        from google.colab import files
        files.download(archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-root', type=Path, default=Path('/content/mira_v2_outputs'))
    parser.add_argument('--print-every', type=int, default=5)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.print_every < 1:
        parser.error('--print-every must be positive')
    run(args.root, args.output_root, args.print_every, args.download)
