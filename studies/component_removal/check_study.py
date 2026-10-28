"""Check immutable study code/input hashes before any fit or saved-result replay.

No model import, fitting, calibration, target inference or output mutation.
"""
from pathlib import Path
import argparse, hashlib, json

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(131072), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Directory with the existing band_streams.npy and manifest.json')
    parser.add_argument('--asset-root', type=Path,
                        help='Optional extracted validated supplement root, containing archives/')
    args = parser.parse_args()
    provenance = json.loads((HERE / 'STUDY-SOURCES.json').read_text())
    for filename, expected in provenance['frozen_files_sha256'].items():
        assert sha(HERE / filename) == expected, f'Frozen file mismatch: {filename}'
    protocol = json.loads((HERE / 'protocol.json').read_text())
    for filename, expected in protocol['implementation_sha256'].items():
        assert sha(HERE / filename) == expected, f'Implementation mismatch: {filename}'
    for filename, expected in protocol['input_sha256'].items():
        assert sha(args.data_root / filename) == expected, f'Input mismatch: {filename}'
    entries = json.loads((args.data_root / 'manifest.json').read_text())['selected']
    assert len(entries) == 1260
    for fold in protocol['folds']:
        tr = [e for e in entries if e['environment'] != fold and e['role'] == 'train']
        va = [e for e in entries if e['environment'] != fold and e['role'] == 'validation']
        assert len(tr) == 504 and len(va) == 168
        assert {e['subject'] for e in tr}.isdisjoint({e['subject'] for e in va})
    archives = 0
    if args.asset_root:
        for relative, expected in provenance['supplement_archive_sha256'].items():
            assert sha(args.asset_root / relative) == expected, f'Archive mismatch: {relative}'
            archives += 1
    print(json.dumps({'passed': True, 'frozen_files_checked': len(provenance['frozen_files_sha256']),
                      'input_hashes_checked': 2, 'outer_source_partitions_checked': 3,
                      'supplement_archives_checked': archives,
                      'training_runs': 0, 'target_inference': 0}, indent=2))


if __name__ == '__main__':
    main()
