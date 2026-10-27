"""Offline inference for a trusted local MIRA-CSI research checkpoint."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from mira_model import ActivityCNN
from missingness import repair,gap_distance

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--input',required=True,help='An amplitude array with shape (18,500) in the audited preprocessing scale')
    parser.add_argument('--observed',help='Optional Boolean array of shape (500,), true for observed positions')
    parser.add_argument('--imputer',choices=['linear','nearest','source_mean'],default='linear')
    parser.add_argument('--calibration',help='Matching results/runs JSON for this checkpoint')
    args=parser.parse_args()
    # Only load checkpoints produced by this project from a trusted source.
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=ActivityCNN(mask_aware=state['variant']=='mira_csi')
    model.load_state_dict(state['state_dict']);model.eval()
    x=np.load(args.input,allow_pickle=False).astype(np.float32)
    assert x.shape==(18,500),'Expected 18 bands x 500 normalized packet positions'
    observed=(np.load(args.observed,allow_pickle=False).astype(bool) if args.observed
              else np.isfinite(x).all(0))
    assert observed.shape==(500,) and observed.any()
    assert np.isfinite(x[:,observed]).all()
    mean=state['mean'];scale=state['scale']
    x=repair(x,observed,args.imputer,mean.ravel())
    temperature=1.
    if args.calibration:
        metadata=json.loads(Path(args.calibration).read_text())
        # The caller must select the matching run metadata, named by model,
        # excluded environment and training seed.
        expected=f"{state['variant']}_E{state['target_environment']}_s{state['seed']}"
        assert Path(args.calibration).stem==expected,'Calibration belongs to another run'
        temperature=float(metadata['temperature'])
    with torch.inference_mode():
        logits=model(torch.from_numpy((x[None]-mean)/scale),
                     torch.from_numpy(observed.astype(np.float32)[None,None]),
                     torch.from_numpy(gap_distance(observed)[None,None]))
        p=torch.softmax(logits.double()/temperature,dim=1).numpy()[0]
    result=dict(prediction=state['classes'][int(p.argmax())],
                probabilities={name:float(v) for name,v in zip(state['classes'],p)},
                observed_fraction=float(observed.mean()),temperature=temperature,
                model=state['variant'],training_seed=state['seed'],
                excluded_environment=state['target_environment'],
                scope='Offline seven-class research prediction; no body image or real-time sensing claim.')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
