"""Offline inference for a trusted local MIRA-CSI research checkpoint."""
from pathlib import Path
import argparse,json,warnings
import numpy as np
import torch
from mira_model import ActivityCNN
from missingness import repair,gap_distance
from integrity import (IntegrityError, calibration_scope, read_json,
                       validate_temperature, verify_checkpoint_metadata)

ROOT=Path(__file__).resolve().parent


def load_observation_mask(path, x):
    if path is None:
        observed=np.isfinite(x).all(0)
    else:
        raw=np.load(path,allow_pickle=False)
        if raw.shape!=(500,): raise IntegrityError('Observation mask must have shape (500,)')
        numeric=(np.issubdtype(raw.dtype,np.bool_) or np.issubdtype(raw.dtype,np.integer)
                 or np.issubdtype(raw.dtype,np.floating))
        if not numeric or not np.isfinite(raw).all() or not ((raw==0)|(raw==1)).all():
            raise IntegrityError('Observation mask must be Boolean or finite binary 0/1')
        observed=raw.astype(bool)
    if observed.shape!=(500,) or not observed.any():
        raise IntegrityError('Observation mask must contain at least one observed position')
    return observed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--input',required=True,help='An amplitude array with shape (18,500) in the audited preprocessing scale')
    parser.add_argument('--observed',help='Optional Boolean array of shape (500,), true for observed positions')
    parser.add_argument('--imputer',choices=['linear','nearest','source_mean'],default='linear')
    parser.add_argument('--calibration',help='Matching results/runs JSON for this checkpoint')
    parser.add_argument('--allow-legacy-calibration',action='store_true',
                        help='Explicitly allow trusted historical metadata not bound to checkpoint bytes; known released pairs are checked automatically')
    args=parser.parse_args()
    # Only load checkpoints produced by this project from a trusted source.
    try:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    except Exception as exc:
        raise IntegrityError('Checkpoint cannot be loaded; it may be incomplete or incompatible') from exc
    verify_checkpoint_metadata(state,ROOT)
    temperature=1.; integrity_scope='uncalibrated; no calibration file supplied'
    if args.calibration:
        metadata=read_json(args.calibration)
        integrity_scope=calibration_scope(metadata,args.calibration,args.checkpoint,state,ROOT,
                                           allow_legacy=args.allow_legacy_calibration)
        temperature=validate_temperature(metadata.get('temperature'))
        if integrity_scope=='legacy_unbound_explicit_opt_in':
            warnings.warn('Explicit legacy opt-in: calibration is not bound to the checkpoint bytes or current source. This is not a verified released pair.',RuntimeWarning)
    x=np.load(args.input,allow_pickle=False).astype(np.float32)
    assert x.shape==(18,500),'Expected 18 bands x 500 normalized packet positions'
    observed=load_observation_mask(args.observed,x)
    assert np.isfinite(x[:,observed]).all()
    mean=state['mean'];scale=state['scale']
    x=repair(x,observed,args.imputer,mean.ravel())
    model=ActivityCNN(mask_aware=state['variant']=='mira_csi')
    model.load_state_dict(state['state_dict']);model.eval()
    with torch.inference_mode():
        logits=model(torch.from_numpy((x[None]-mean)/scale),
                     torch.from_numpy(observed.astype(np.float32)[None,None]),
                     torch.from_numpy(gap_distance(observed)[None,None]))
        if logits.shape!=(1,len(state['classes'])) or not torch.isfinite(logits).all():
            raise IntegrityError('Checkpoint produced nonfinite or invalid-shaped logits')
        p=torch.softmax(logits.double()/temperature,dim=1).numpy()[0]
    if not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1.,atol=1e-12,rtol=1e-12):
        raise IntegrityError('Checkpoint produced an invalid probability vector')
    result=dict(prediction=state['classes'][int(p.argmax())],
                probabilities={name:float(v) for name,v in zip(state['classes'],p)},
                observed_fraction=float(observed.mean()),temperature=temperature,
                model=state['variant'],training_seed=state['seed'],
                excluded_environment=state['target_environment'],
                calibration_integrity=integrity_scope,
                scope='Offline seven-class research prediction; no body image or real-time sensing claim.')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
