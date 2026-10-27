"""Independent recomputation from saved probabilities, with completeness checks."""
from pathlib import Path
import csv, json
import numpy as np
from sklearn.metrics import f1_score

root = Path(__file__).resolve().parent
out = root/'results'; config = json.loads((root/'protocol.json').read_text())
manifest = json.loads((root/'data/manifest.json').read_text())['selected']
conditions = json.loads((out/'conditions.json').read_text())
checked = 0; rows_checked = 0; errors = []
for target in config['folds']:
    for seed in config['training_seeds']:
        for model in config['models']:
            tag = f'{model}_E{target}_s{seed}'
            path = out/'predictions'/f'{tag}.npz'
            if not path.exists():
                errors.append('Missing '+tag); continue
            data = np.load(path); p = data['probabilities']; y = data['truth']
            indices = data['indices']; held = data['held']
            assert p.shape == (31,588,7)
            assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            np.testing.assert_allclose(p.sum(-1),1,atol=1e-12)
            assert len(set(indices.tolist())) == 588
            assert np.array_equal(y,np.array([config['classes'].index(manifest[i]['activity']) for i in indices]))
            assert np.array_equal(held,np.array([manifest[i]['environment']==target for i in indices]))
            rows = list(csv.DictReader((out/'metrics'/f'{tag}.csv').open()))
            assert len(rows) == 62
            for row in rows:
                cid = int(row['condition_id']); mask = held if row['domain']=='held' else ~held
                pp = p[cid,mask]; yy = y[mask]; pred = pp.argmax(-1); correct = pred==yy
                n = 420 if row['domain']=='held' else 168
                assert len(yy) == n == int(row['n'])
                assert int(correct.sum()) == int(row['correct'])
                check = dict(accuracy=correct.mean(),
                    macro_f1=f1_score(yy,pred,labels=list(range(7)),average='macro',zero_division=0),
                    nll=-np.log(np.clip(pp[np.arange(n),yy],np.finfo(float).eps,1)).mean(),
                    brier=np.square(pp-np.eye(7)[yy]).sum(1).mean())
                confidence=pp.max(1); ece=0.
                for b in range(10):
                    binmask=(confidence>=b/10)&((confidence<(b+1)/10) if b<9 else (confidence<=1))
                    if binmask.any():ece+=binmask.mean()*abs(confidence[binmask].mean()-correct[binmask].mean())
                check['ece']=ece
                for key,value in check.items():
                    np.testing.assert_allclose(value,float(row[key]),atol=1e-10,rtol=1e-10,err_msg=tag+' '+key)
                rows_checked+=1
            checked+=1
receipt=dict(complete=(checked==60 and not errors),runs_checked=checked,
             expected_runs=60,metric_rows_checked=rows_checked,
             checks='Saved per-file probabilities: identity, domain, truth, normalization, counts, accuracy, macro F1, NLL, Brier, ECE.',
             missing=errors)
(out/'verification.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
if not receipt['complete']:
    raise SystemExit('Incomplete experiment: no final 60-run claim is allowed.')
