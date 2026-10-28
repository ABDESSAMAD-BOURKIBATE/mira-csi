"""Check published file consistency and saved validation metrics, no inference."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL='cc407a98f343dcac0230c3fe2a733a429e8f980b6748ddfd8240e4542696fc32'
EXPECTED_FREEZE='c7575d16aefd0a12348f0e3997683cb1d6895641ebf5e0c8541780b86be15b5b'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(y,p,lp):
    pred=p.argmax(1)
    matrix=np.zeros((7,7),dtype=np.int64)
    for truth,guess in zip(y,pred):
        matrix[int(truth),int(guess)]+=1
    denom=matrix.sum(0)+matrix.sum(1)
    f1=np.divide(2*matrix.diagonal(),denom,out=np.zeros(7),where=denom>0)
    return dict(accuracy=float((pred==y).mean()),macro_f1=float(f1.mean()),
                nll=float(-lp[np.arange(len(y)),y].mean()))

def verify():
    manifest=json.loads((ROOT/'release-manifest.json').read_text())
    for relative,expected in manifest['files_sha256'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or digest(path)!=expected:
            raise ValueError('File hash check failed: '+relative)
    assert digest(ROOT/'frozen/protocol.json')==EXPECTED_PROTOCOL
    assert digest(ROOT/'artifacts/source_fit_freeze.json')==EXPECTED_FREEZE
    protocol=json.loads((ROOT/'frozen/protocol.json').read_text())
    freeze=json.loads((ROOT/'artifacts/source_fit_freeze.json').read_text())
    calibration=json.loads((ROOT/'artifacts/calibration-index.json').read_text())
    assert freeze['completed_source_fits']==75 and len(freeze['fits'])==75
    for name,expected in protocol['implementation_sha256'].items():
        assert digest(ROOT/'frozen'/name)==expected
    counted=0
    for fold in (1,2,3):
        for seed in range(20261026,20261031):
            tag=f'E_E{fold}_s{seed}'
            binding=freeze['fits'][tag]
            assert digest(ROOT/'artifacts/checkpoints'/f'{tag}.pt')==binding['checkpoint_sha256']
            assert digest(ROOT/'artifacts/validation'/f'{tag}.npz')==binding['validation_sha256']
            metadata_path=ROOT/'artifacts/runs'/f'{tag}.training.json'
            assert digest(metadata_path)==binding['receipt_sha256']
            meta=json.loads(metadata_path.read_text())
            assert calibration[tag]['classes']==protocol['classes']
            assert calibration[tag]['temperature']==binding['temperature']==meta['temperature']
            assert calibration[tag]['checkpoint_sha256']==binding['checkpoint_sha256']
            with np.load(ROOT/'artifacts/validation'/f'{tag}.npz',allow_pickle=False) as val:
                p,lp=val['probabilities'],val['log_probabilities']
                z=val['logits'].astype(np.float64)/float(val['temperature'])
                centered=z-z.max(axis=1,keepdims=True)
                expected_lp=centered-np.log(np.exp(centered).sum(axis=1,keepdims=True))
                np.testing.assert_allclose(lp,expected_lp,atol=1e-12,rtol=0)
                np.testing.assert_allclose(p,np.exp(lp),atol=1e-12,rtol=0)
                found=metrics(val['truth'],p,lp)
                for key,value in found.items():
                    np.testing.assert_allclose(value,meta['source_validation_'+key],atol=1e-12,rtol=0)
            counted+=1
    return dict(passed=True,verified_files=len(manifest['files_sha256']),verified_E_weights=counted,
                source_validation_vectors=counted*168,model_forward_passes=0,
                integrity_note='Hash consistency is not proof of authenticity or external validity.')

if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
