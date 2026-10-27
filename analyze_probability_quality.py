"""Descriptive post-hoc probability/ranking audit of already frozen predictions.

No model, temperature or operating threshold is fitted in this analysis.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent;out=root/'results'
assert json.loads((out/'verification.json').read_text())['complete']
conditions=json.loads((out/'conditions.json').read_text())
models=['cnn_clean','cnn_augmented','mira_csi','minirocket_9996']
labels=['CNN','CNN + augmentation','MIRA-CSI','MiniRocket (9,996)']
rows=[];thresholds=[]
for model in models:
    for seed in range(20261026,20261031):
        for pattern in ['complete','contiguous']:
            cid=next(i for i,c in enumerate(conditions) if c[0]==pattern and
                     (pattern=='complete' or (c[2]==seed and c[3]=='linear')))
            truth=[];prob=[]
            for fold in [1,2,3]:
                data=np.load(out/'predictions'/f'{model}_E{fold}_s{seed}.npz')
                ids=data['condition_ids'] if 'condition_ids' in data.files else np.arange(31)
                position=int(np.flatnonzero(ids==cid)[0]);held=data['held']
                truth.append(data['truth'][held]);prob.append(data['probabilities'][position,held])
            y=np.concatenate(truth);p=np.concatenate(prob)
            assert len(y)==1260 and np.array_equal(np.bincount(y),np.full(7,180))
            error=(p.argmax(1)!=y);confidence=p.max(1)
            order=np.argsort(-confidence,kind='stable')
            for numerator in range(1,11):
                accepted=numerator*126
                rows.append(dict(model=model,training_seed=seed,pattern=pattern,
                                 coverage=accepted/1260,accepted=accepted,n=1260,
                                 selective_error=float(error[order[:accepted]].mean())))
            for threshold in [.5,.8]:
                keep=confidence>=threshold
                thresholds.append(dict(model=model,training_seed=seed,pattern=pattern,
                                       threshold=threshold,accepted=int(keep.sum()),n=1260,
                                       coverage=float(keep.mean()),
                                       selective_error=float(error[keep].mean()) if keep.any() else None))
df=pd.DataFrame(rows);df.to_csv(out/'neural_risk_coverage_per_seed.csv',index=False)
pd.DataFrame(thresholds).to_csv(out/'neural_fixed_thresholds.csv',index=False)
summary=df.groupby(['model','pattern','coverage'],as_index=False).selective_error.agg(['mean','std']).reset_index()
summary.to_csv(out/'neural_risk_coverage_summary.csv',index=False)
uniform=dict(classes=7,probability_per_class=1/7,nll=float(np.log(7)),brier=6/7,
             scope='Proper-score reference only, on balanced held sets. It supplies no useful class discrimination; no risk-coverage curve is inferred from tied uniform probabilities.')
(out/'uniform_reference.json').write_text(json.dumps(uniform,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,4,figsize=(11.5,3.5),sharey=True)
for ax,model,label in zip(axes,models,labels):
    for pattern,color,marker,title in [('complete','#456A9C','o','Complete'),('contiguous','#BE7B22','s','Block 50%')]:
        a=summary[(summary.model==model)&(summary.pattern==pattern)].sort_values('coverage')
        x=a.coverage.to_numpy();mean=a['mean'].to_numpy();sd=a['std'].to_numpy()
        ax.plot(x,mean,color=color,marker=marker,markersize=3,lw=1.3,label=title)
        ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=color,alpha=.13,lw=0)
    ax.set_title(label,fontsize=9);ax.set_xlabel('Retained fraction');ax.set_xlim(.1,1);ax.set_ylim(0,1)
    ax.set_xticks([.2,.4,.6,.8,1]);ax.grid(alpha=.16)
axes[0].set_ylabel('Error among retained predictions');axes[-1].legend(fontsize=8,loc='lower right')
fig.text(.5,.01,'Mean ± seed SD; confidence-ranked within fixed held tests. Descriptive ranking, not a validated deployment threshold.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.055,1,1));fig.savefig(out/'neural_risk_coverage.pdf');fig.savefig(out/'neural_risk_coverage.png',dpi=200);plt.close(fig)
audit=dict(status='complete',analysis='Post-hoc descriptive audit of frozen predictions',
           risk_rows=len(df),threshold_rows=len(thresholds),uniform_reference=uniform,
           no_new_model_fitting=True,no_new_calibration=True)
(out/'probability_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
