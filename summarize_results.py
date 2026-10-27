"""Summaries and publication plots computed only from verified result files."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent; out=root/'results'
receipt=json.loads((out/'verification.json').read_text())
assert receipt['complete'], 'Run independent verification first.'
df=pd.concat([pd.read_csv(p) for p in sorted((out/'metrics').glob('*.csv'))],ignore_index=True)
assert len(df)==3720
df.to_csv(out/'all_metrics.csv',index=False)
measures=['accuracy','macro_f1','nll','brier','ece']
# Each row has the same number of files within a domain. Average the three
# folds and mask realizations separately per training seed. These folds are
# not treated as independent participants or confidence-interval replicates.
per_seed=df.groupby(['model','training_seed','domain','pattern','imputer'],as_index=False)[measures].mean()
per_seed.to_csv(out/'per_seed.csv',index=False)
summary=per_seed.groupby(['model','domain','pattern','imputer'])[measures].agg(['mean','std'])
summary.columns=['_'.join(c) for c in summary.columns]
summary=summary.reset_index(); summary.to_csv(out/'summary.csv',index=False)
folds=df.groupby(['model','held_environment','domain','pattern','imputer'],as_index=False)[measures].mean()
folds.to_csv(out/'per_fold.csv',index=False)
models=['cnn_clean','cnn_augmented','mira_csi','minirocket_9996']
labels=['CNN','CNN + loss augmentation','MIRA-CSI','MiniRocket (9,996)']
colors=['#456A9C','#C37A24','#007E7B','#805BA0']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10.2,3.9),sharey=True)
for ax,domain,title in zip(axes,['familiar','held'],['Familiar source environments','Excluded environment + cohort']):
    for j,(model,label,color) in enumerate(zip(models,labels,colors)):
        subset=per_seed[(per_seed.model==model)&(per_seed.domain==domain)]
        values=[]
        for pattern,method in [('complete','none'),('random','linear'),('contiguous','linear')]:
            v=subset[(subset.pattern==pattern)&(subset.imputer==method)].accuracy.to_numpy()*100
            assert len(v)==5; values.append(v)
        avg=np.array([v.mean() for v in values]); sd=np.array([v.std(ddof=1) for v in values])
        ax.errorbar(np.arange(3)+(j-1.5)*.055,avg,yerr=sd,label=label,color=color,
                    marker=['o','s','D','^'][j],capsize=3,lw=1.5)
    ax.set_xticks(range(3),['Complete','Random 50%','Block 50%']); ax.set_title(title)
    ax.set_ylim(0,100); ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Accuracy (%)'); axes[1].legend(fontsize=8,loc='upper right')
fig.text(.5,-.02,'Points: mean of five training seeds; bars: sample SD across seeds.\nEach corrupted point averages five matched masks and three folds; linear repair.',ha='center',fontsize=9)
fig.tight_layout(); fig.savefig(out/'expanded_comparison.pdf'); fig.savefig(out/'expanded_comparison.png',dpi=200); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10.2,4),sharey=True)
for ax,pattern,title in zip(axes,['random','contiguous'],['Random 50% missingness','Contiguous 50% missingness']):
    for j,(model,label,color) in enumerate(zip(models,labels,colors)):
        subset=per_seed[(per_seed.model==model)&(per_seed.domain=='held')&(per_seed.pattern==pattern)]
        v=[subset[subset.imputer==method].macro_f1.to_numpy() for method in ['linear','nearest','source_mean']]
        ax.errorbar(np.arange(3)+(j-1.5)*.055,[x.mean() for x in v],
                    yerr=[x.std(ddof=1) for x in v],marker=['o','s','D','^'][j],label=label,
                    color=color,lw=1.5,capsize=3)
    ax.set_xticks(range(3),['Linear','Nearest','Source mean']); ax.set_title(title); ax.grid(axis='y',alpha=.2)
    ax.set_ylim(0,1)
axes[0].set_ylabel('Held-environment macro F1'); axes[1].legend(fontsize=8)
fig.text(.5,-.01,'All methods retain observed samples. Means ± SD over five training seeds; five masks per seed.',ha='center',fontsize=9)
fig.tight_layout(); fig.savefig(out/'imputation_comparison.pdf'); fig.savefig(out/'imputation_comparison.png',dpi=200); plt.close(fig)

changes=[]
for pattern in ['complete','random','contiguous']:
    method='none' if pattern=='complete' else 'linear'
    selection=per_seed[(per_seed.domain=='held')&(per_seed.pattern==pattern)&(per_seed.imputer==method)]
    wide=selection.pivot(index='training_seed',columns='model',values='macro_f1')
    for a,b in [('cnn_augmented','cnn_clean'),('mira_csi','cnn_augmented')]:
        delta=wide[a]-wide[b]
        changes.append(dict(pattern=pattern,comparison=a+' minus '+b,mean_macro_f1_delta=delta.mean(),
                            sd_macro_f1_delta=delta.std(ddof=1),positive_seeds=int((delta>0).sum()),seeds=5))
pd.DataFrame(changes).to_csv(out/'ablation_deltas.csv',index=False)
print(summary[(summary.domain=='held')&((summary.pattern=='complete')|((summary.pattern=='contiguous')&(summary.imputer=='linear')))].to_string(index=False))
print('Saved verified summaries, per-fold results and two publication figure pairs.')
