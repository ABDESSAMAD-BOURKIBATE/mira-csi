"""Summaries and publication plots computed only from verified result files."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from missingness import make_mask,repair

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-root',type=Path,default=root/'results')
args=parser.parse_args();out=args.output_root.resolve()
receipt=json.loads((out/'verification.json').read_text())
assert receipt['complete'], 'Run independent verification first.'
df=pd.concat([pd.read_csv(p) for p in sorted((out/'metrics').glob('*.csv'))],ignore_index=True)
df.to_csv(out/'all_metrics.csv',index=False)
measures=['accuracy','macro_f1','nll','brier','ece']
# Each row has the same number of files within a domain. Average the three
# folds and mask realizations separately per training seed. These folds are
# not treated as independent participants or confidence-interval replicates.
primary=df[(df.pattern=='complete')|(df.mask_seed==df.training_seed)].copy()
assert len(primary)==840  # 60 fitted runs x 7 matched conditions x 2 domains.
primary.to_csv(out/'primary_metrics.csv',index=False)
def pooled_scores(y,p):
    correct=p.argmax(1)==y;confidence=p.max(1);ece=0.
    for b in range(10):
        keep=(confidence>=b/10)&((confidence<(b+1)/10) if b<9 else (confidence<=1))
        if keep.any():ece+=keep.mean()*abs(confidence[keep].mean()-correct[keep].mean())
    return dict(accuracy=correct.mean(),macro_f1=f1_score(y,p.argmax(1),labels=range(7),average='macro',zero_division=0),
                nll=-np.log(np.clip(p[np.arange(len(y)),y],np.finfo(float).eps,1)).mean(),
                brier=np.square(p-np.eye(7)[y]).sum(1).mean(),ece=ece)
config=json.loads((root/'protocol.json').read_text())
conditions=json.loads((out/'conditions.json').read_text());pooled=[]
for model in config['models']:
    for seed in config['training_seeds']:
        cases=[(i,c) for i,c in enumerate(conditions) if c[1]==0 or c[2]==seed]
        archives=[np.load(out/'predictions'/f'{model}_E{fold}_s{seed}.npz') for fold in [1,2,3]]
        for domain in ['held','familiar']:
            for cid,(pattern,fraction,mask_seed,method) in cases:
                truths=[];probability=[]
                for archive in archives:
                    ids=archive['condition_ids'] if 'condition_ids' in archive.files else np.arange(31)
                    position=int(np.flatnonzero(ids==cid)[0])
                    keep=archive['held'] if domain=='held' else ~archive['held']
                    truths.append(archive['truth'][keep]);probability.append(archive['probabilities'][position,keep])
                yy=np.concatenate(truths);pp=np.concatenate(probability)
                pooled.append(dict(model=model,training_seed=seed,domain=domain,pattern=pattern,imputer=method,
                                   **pooled_scores(yy,pp)))
per_seed=pd.DataFrame(pooled)
per_seed.to_csv(out/'per_seed.csv',index=False)
summary=per_seed.groupby(['model','domain','pattern','imputer'])[measures].agg(['mean','std'])
summary.columns=['_'.join(c) for c in summary.columns]
summary=summary.reset_index(); summary.to_csv(out/'summary.csv',index=False)
folds=primary.groupby(['model','held_environment','domain','pattern','imputer'],as_index=False)[measures].mean()
folds.to_csv(out/'per_fold.csv',index=False)

# Score repairs against the truly hidden values only after fitting decisions.
# This is a synthetic-mask reconstruction test, not unavailable live ground truth.
entries=json.loads((root/'data/manifest.json').read_text())['selected']
X=np.load(root/'data/band_streams.npy',mmap_mode='r')
env=np.array([e['environment'] for e in entries]);roles=np.array([e['role'] for e in entries])
reconstruction=[]
for fold in [1,2,3]:
    source=(env!=fold)&(roles=='train');mean=np.asarray(X[source]).mean((0,2))
    for seed in config['training_seeds']:
        for pattern in ['random','contiguous']:
            for method in config['imputers']:
                sq=0.;ab=0.;count=0
                for i in np.flatnonzero(env==fold):
                    observed=make_mask(entries[i]['name'],pattern,.5,seed)
                    recovered=repair(np.array(X[i]),observed,method,mean)
                    delta=(recovered[:,~observed]-X[i,:,~observed].T).astype(np.float64)
                    sq+=np.square(delta).sum();ab+=np.abs(delta).sum();count+=delta.size
                reconstruction.append(dict(held_environment=fold,mask_seed=seed,pattern=pattern,imputer=method,
                                           hidden_values=count,squared_error_sum=sq,absolute_error_sum=ab,
                                           rmse_db=np.sqrt(sq/count),mae_db=ab/count))
reconstruction=pd.DataFrame(reconstruction);reconstruction.to_csv(out/'imputation_reconstruction.csv',index=False)
repaired=[]
for (pattern,method),g in reconstruction.groupby(['pattern','imputer']):
    repaired.append(dict(pattern=pattern,imputer=method,
                         rmse_db=np.sqrt(g.squared_error_sum.sum()/g.hidden_values.sum()),
                         mae_db=g.absolute_error_sum.sum()/g.hidden_values.sum()))
pd.DataFrame(repaired).to_csv(out/'imputation_reconstruction_summary.csv',index=False)
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
fig.text(.5,-.02,'Points: mean of five paired training/mask seeds; bars: sample SD.\nEach seed pools three folds; linear repair. Familiar files recur across folds.',ha='center',fontsize=9)
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
fig.text(.5,-.01,'Observed samples retained. Means ± SD over five paired training/mask seeds; three folds per seed.',ha='center',fontsize=9)
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
