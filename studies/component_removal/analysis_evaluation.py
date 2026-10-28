"""Independent post-hoc analysis of frozen MIRA component/ResNet evaluations.

Reads archives without extracting over or modifying the source-fit freeze.
No runner import, model fitting, calibration fitting, or outcome-based selection.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,time,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
PROTOCOL_SHA='2de83950b8aa40b601b11976ca4c2eeadb77cf9bd054fb4fedfac2147d7a8231'
VARIANTS=['mira_full','no_mask_input','no_distance_input','uniform_pool','resnet1d_clean','resnet1d_augmented']
LABELS={'mira_full':'Full MIRA','no_mask_input':'No mask input','no_distance_input':'No distance input','uniform_pool':'Uniform pool','resnet1d_clean':'ResNet clean','resnet1d_augmented':'ResNet augmented'}
COLORS=['#00796B','#D55E00','#0072B2','#9C609D','#58666E','#B49100']
COMPARATORS=['no_mask_input','no_distance_input','uniform_pool']
GROUPS=[('complete','none')]+[(p,m) for p in ['random','contiguous'] for m in ['linear','nearest','source_mean']]
METRICS=['accuracy','macro_f1','nll','brier','ece']
BOOT_METRICS=['accuracy','macro_f1','nll','brier']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class Archive:
    def __init__(self,path):
        self.path=Path(path).resolve();self.zip=zipfile.ZipFile(self.path);self.names=self.zip.namelist()
    def member(self,relative,required=True):
        matches=[n for n in self.names if n==relative or n.endswith('/'+relative)]
        if not matches and not required:return None
        if len(matches)!=1:raise ValueError(f'{self.path.name}: expected one {relative}, found {matches}')
        return matches[0]
    def read(self,relative):return self.zip.read(self.member(relative))
    def json(self,relative):return json.loads(self.read(relative))
    def close(self):self.zip.close()


def score(y,p):
    pred=p.argmax(1);correct=pred==y;confidence=p.max(1)
    cm=np.bincount(y*7+pred,minlength=49).reshape(7,7)
    denom=cm.sum(0)+cm.sum(1)
    macro=np.divide(2*np.diag(cm),denom,out=np.zeros(7),where=denom>0).mean()
    nll=-np.log(np.clip(p[np.arange(len(y)),y],np.finfo(float).eps,1)).mean()
    brier=np.square(p-np.eye(7)[y]).sum(1).mean()
    bins=np.searchsorted(np.arange(1,10)/10,confidence,side='right')
    ece=np.abs(np.bincount(bins,weights=correct,minlength=10)-np.bincount(bins,weights=confidence,minlength=10)).sum()/len(y)
    return dict(n=len(y),correct=int(correct.sum()),accuracy=float(correct.mean()),macro_f1=float(macro),nll=float(nll),brier=float(brier),ece=float(ece))


def cm_scores(cm):
    diag=np.diagonal(cm,axis1=-2,axis2=-1)
    denom=cm.sum(-1)+cm.sum(-2)
    f1=np.divide(2*diag,denom,out=np.zeros_like(diag,dtype=float),where=denom>0).mean(-1)
    accuracy=diag.sum(-1)/cm.sum((-2,-1))
    return np.stack([accuracy,f1],axis=-1)


def group_ids(conditions,pattern,imputer):
    result=[i for i,c in enumerate(conditions) if c[0]==pattern and c[3]==imputer]
    assert len(result)==(1 if pattern=='complete' else 5)
    return result


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def validate_and_pool(source,evaluation,config,entries,out):
    seeds=config['training_seeds'];truth=np.array([config['classes'].index(e['activity']) for e in entries])
    conditions=[['complete',0.,0,'none']]+[[p,.5,s,m] for p in ['random','contiguous'] for m in config['imputers'] for s in config['mask_seeds']]
    assert evaluation.json('conditions.json')==conditions==source.json('conditions.json')
    runtime=source.json('runtime.json');assert runtime['protocol_sha256']==PROTOCOL_SHA and runtime['target_evaluation_requested'] is False
    assert not any(n.endswith('.evaluation.json') for n in source.names)
    assert not any('/predictions/' in n and n.endswith('.npz') for n in source.names)
    pools={v:np.empty((5,31,1260,7),np.float64) for v in VARIANTS}
    counts={v:np.zeros((5,1260),np.uint8) for v in VARIANTS}
    rows=[];source_hashes={};prediction_hashes={};max_error=0.;eval_weights_verified=0;count=0
    freeze_manifest_name=evaluation.member('evaluation-freeze-manifest.json',required=False)
    freeze_manifest=(json.loads(evaluation.zip.read(freeze_manifest_name)) if freeze_manifest_name else None)
    if freeze_manifest:assert freeze_manifest['protocol_sha256']==PROTOCOL_SHA
    for v in VARIANTS:
        for si,seed in enumerate(seeds):
            for fold in [1,2,3]:
                tag=f'{v}_E{fold}_s{seed}'
                meta=source.json(f'runs/{tag}.training.json')
                assert meta['protocol_sha256']==PROTOCOL_SHA and meta['mode']=='completed_source_training'
                assert meta['variant']==v and meta['training_seed']==seed and meta['held_environment']==fold
                raw_checkpoint=source.read(f'checkpoints/{tag}.pt')
                checkpoint_hash=hashlib.sha256(raw_checkpoint).hexdigest()
                assert checkpoint_hash==meta['checkpoint_sha256'];source_hashes[tag]=checkpoint_hash
                # An evaluation package must retain the same source training metadata.
                assert evaluation.json(f'runs/{tag}.training.json')==meta
                evaluation_checkpoint=evaluation.member(f'checkpoints/{tag}.pt',required=False)
                if evaluation_checkpoint:
                    assert hashlib.sha256(evaluation.zip.read(evaluation_checkpoint)).hexdigest()==checkpoint_hash
                    eval_weights_verified+=1
                elif freeze_manifest:
                    assert freeze_manifest['checkpoint_sha256_after_evaluation'][tag]==checkpoint_hash
                    eval_weights_verified+=1
                ep=evaluation.json(f'runs/{tag}.evaluation.json');assert ep['protocol_sha256']==PROTOCOL_SHA and ep['target_scored'] is True
                raw=evaluation.read(f'predictions/{tag}.npz');ph=hashlib.sha256(raw).hexdigest()
                assert ep['prediction_sha256']==ph;prediction_hashes[tag]=ph
                z=np.load(io.BytesIO(raw),allow_pickle=False)
                p=z['probabilities'];y=z['truth'];ix=z['indices'];held=z['held']
                np.testing.assert_array_equal(z['condition_ids'],np.arange(31))
                fam=np.array([i for i,e in enumerate(entries) if e['environment']!=fold and e['role']=='familiar_test'])
                target=np.array([i for i,e in enumerate(entries) if e['environment']==fold])
                np.testing.assert_array_equal(ix,np.r_[fam,target]);np.testing.assert_array_equal(y,truth[ix])
                np.testing.assert_array_equal(held,[entries[i]['environment']==fold for i in ix])
                assert p.shape==(31,588,7) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                np.testing.assert_allclose(p.sum(2),1,atol=1e-12)
                gix=ix[held];pools[v][si][:,gix,:]=p[:,held,:];counts[v][si,gix]+=1
                saved=pd.read_csv(io.BytesIO(evaluation.read(f'metrics/{tag}.csv')))
                assert len(saved)==62 and not saved.duplicated(['condition_id','domain']).any()
                saved=saved.set_index(['condition_id','domain'])
                for cid,(pattern,fraction,mask_seed,imputer) in enumerate(conditions):
                    for domain,keep in [('held',held),('familiar',~held)]:
                        result=score(y[keep],p[cid,keep]);original=saved.loc[(cid,domain)]
                        assert original['pattern']==pattern and original['imputer']==imputer and int(original['mask_seed'])==mask_seed
                        for k,value in result.items():
                            error=abs(float(value)-float(original[k]));max_error=max(max_error,error)
                            assert error<1e-10,(tag,cid,domain,k,error)
                        rows.append(dict(model=v,training_seed=seed,held_environment=fold,domain=domain,condition_id=cid,pattern=pattern,missing_fraction=fraction,mask_seed=mask_seed,imputer=imputer,**result))
                count+=1
                if count%15==0:print(f'Independently verified frozen probabilities/metrics and source checkpoint hashes: {count}/90',flush=True)
    assert count==90 and len(rows)==5580
    for v in VARIANTS:assert np.all(counts[v]==1)
    pd.DataFrame(rows).to_csv(out/'verified-per-fit-metrics.csv',index=False)
    if freeze_manifest:
        assert set(freeze_manifest['prediction_sha256'])==set(prediction_hashes)
        for tag,ph in prediction_hashes.items():assert freeze_manifest['prediction_sha256'][tag]==ph
    receipt=dict(fits=90,conditions_per_fit=31,metric_rows=5580,probability_vectors=90*31*588,
        max_absolute_saved_metric_difference=max_error,source_checkpoint_hashes_verified=90,
        evaluation_checkpoint_hashes_or_post_eval_manifest_verified=eval_weights_verified,
        source_checkpoint_sha256=source_hashes,prediction_sha256=prediction_hashes,
        frozen_weights_limit=('All evaluation checkpoint bytes/post-evaluation hashes match the immutable source freeze.' if eval_weights_verified==90 else 'Evaluation probabilities and source metadata match, but some post-evaluation checkpoint bytes/hashes were not packaged; no stronger local verification of those bytes is claimed.'))
    return pools,truth,conditions,receipt


def summarise(pools,truth,conditions,entries,out):
    people=sorted({(e['environment'],e['subject']) for e in entries});assert len(people)==30
    lookup={p:i for i,p in enumerate(people)}
    persons=np.array([lookup[(e['environment'],e['subject'])] for e in entries])
    strata=[np.array([j for j,p in enumerate(people) if p[0]==f]) for f in [1,2,3]]
    assert all(len(s)==10 for s in strata) and np.all(np.bincount(persons)==42)
    cell_rows=[];perfold=[];participant={};raw_confusions={}
    seeds=list(range(20261026,20261031))
    for v in VARIANTS:
        cms=np.zeros((5,31,30,7,7),np.int16);nll=np.zeros((5,31,30));brier=np.zeros_like(nll)
        for si,seed in enumerate(seeds):
            for cid,(pattern,fraction,mask_seed,imputer) in enumerate(conditions):
                p=pools[v][si,cid];pred=p.argmax(1)
                result=score(truth,p)
                cell_rows.append(dict(model=v,training_seed=seed,condition_id=cid,pattern=pattern,imputer=imputer,mask_seed=mask_seed,**result))
                cms[si,cid]=np.bincount(persons*49+truth*7+pred,minlength=30*49).reshape(30,7,7)
                nll_each=-np.log(np.clip(p[np.arange(1260),truth],np.finfo(float).eps,1))
                brier_each=np.square(p-np.eye(7)[truth]).sum(1)
                nll[si,cid]=np.bincount(persons,weights=nll_each,minlength=30)
                brier[si,cid]=np.bincount(persons,weights=brier_each,minlength=30)
                for fold in [1,2,3]:
                    keep=np.array([e['environment']==fold for e in entries])
                    perfold.append(dict(model=v,training_seed=seed,held_environment=fold,condition_id=cid,pattern=pattern,imputer=imputer,mask_seed=mask_seed,**score(truth[keep],p[keep])))
        assert np.all(cms.sum((-2,-1))==42)
        participant[v]=(cms,nll,brier)
        raw_confusions[v]=np.stack([cms[:,:,s].sum(2) for s in strata])
    cells=pd.DataFrame(cell_rows);assert len(cells)==930
    cells.to_csv(out/'pooled-held-per-training-mask.csv',index=False)
    per_seed=cells.groupby(['model','training_seed','pattern','imputer'],sort=False)[METRICS].mean().reset_index()
    assert len(per_seed)==210;per_seed.to_csv(out/'pooled-held-mask-averaged-per-seed.csv',index=False)
    summary=per_seed.groupby(['model','pattern','imputer'],sort=False)[METRICS].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index()
    assert len(summary)==42;summary.to_csv(out/'held-summary.csv',index=False)
    pf=pd.DataFrame(perfold)
    pf_seed=pf.groupby(['model','held_environment','training_seed','pattern','imputer'],sort=False)[METRICS].mean().reset_index()
    pf_seed.to_csv(out/'per-environment-mask-averaged-per-seed.csv',index=False)
    pf_summary=pf_seed.groupby(['model','held_environment','pattern','imputer'],sort=False)[METRICS].agg(['mean','std'])
    pf_summary.columns=['_'.join(x) for x in pf_summary.columns];pf_summary=pf_summary.reset_index()
    pf_summary.to_csv(out/'per-environment-summary.csv',index=False)
    np.savez_compressed(out/'per-environment-confusions-all-cells.npz',**raw_confusions)
    confusion_rows=[]
    for v in VARIANTS:
        for fi,fold in enumerate([1,2,3]):
            for pattern,imputer in GROUPS:
                ids=group_ids(conditions,pattern,imputer)
                c=raw_confusions[v][fi][:,ids].mean((0,1));den=c.sum(1,keepdims=True)
                rate=np.divide(c,den,out=np.zeros_like(c),where=den>0)
                for y in range(7):
                    for pred in range(7):confusion_rows.append(dict(model=v,held_environment=fold,pattern=pattern,imputer=imputer,true_class=y,predicted_class=pred,mean_count_per_cell=c[y,pred],mean_row_fraction=rate[y,pred]))
    pd.DataFrame(confusion_rows).to_csv(out/'per-environment-confusions-summary.csv',index=False)
    return summary,pf_summary,participant,people,strata,raw_confusions


def bootstrap_analysis(participant,conditions,people,strata,out,draws,seed,indices_path=None):
    if indices_path:
        z=np.load(indices_path,allow_pickle=False);weights=z['weights'];np.testing.assert_array_equal(z['people'],np.array(people))
        assert weights.shape==(draws,30)
    else:
        rng=np.random.default_rng(seed);weights=np.zeros((draws,30),np.int16)
        for s in strata:weights[:,s]=rng.multinomial(10,np.full(10,.1),size=draws)
    assert np.all(weights>=0) and all(np.all(weights[:,s].sum(1)==10) for s in strata)
    np.savez_compressed(out/'participant-bootstrap-indices.npz',weights=weights,people=np.array(people),seed=seed)
    points={};samples={}
    for v in ['mira_full',*COMPARATORS]:
        cms,nll,brier=participant[v]
        for pattern,imputer in GROUPS:
            ids=group_ids(conditions,pattern,imputer)
            cc=cms[:,ids].reshape(-1,30,7,7).astype(np.float64)
            nn=nll[:,ids].reshape(-1,30);bb=brier[:,ids].reshape(-1,30)
            points[(v,pattern,imputer)]=np.r_[cm_scores(cc.sum(1)).mean(0),nn.sum(1).mean()/1260,bb.sum(1).mean()/1260]
            result=[]
            for begin in range(0,draws,250):
                w=weights[begin:begin+250]
                mixed=np.einsum('bp,cpij->bcij',w,cc,optimize=True)
                af=cm_scores(mixed).mean(1)
                nl=np.einsum('bp,cp->bc',w,nn,optimize=True).mean(1)/1260
                br=np.einsum('bp,cp->bc',w,bb,optimize=True).mean(1)/1260
                result.append(np.column_stack([af,nl,br]))
            samples[(v,pattern,imputer)]=np.concatenate(result)
        print('Participant bootstrap completed for',v,flush=True)
    rows=[];stored={}
    for comparator in COMPARATORS:
        for pattern,imputer in GROUPS:
            lhs=('mira_full',pattern,imputer);rhs=(comparator,pattern,imputer)
            delta=points[lhs]-points[rhs];rep=samples[lhs]-samples[rhs]
            stored[f'{comparator}|{pattern}|{imputer}']=rep
            for j,metric in enumerate(BOOT_METRICS):
                low,high=np.percentile(rep[:,j],[2.5,97.5]);primary=pattern=='contiguous' and imputer=='linear' and metric=='macro_f1'
                adjusted=(np.percentile(rep[:,j],[100*.05/(2*3),100*(1-.05/(2*3))]) if primary else [np.nan,np.nan])
                if primary:assert adjusted[0]<=low<=high<=adjusted[1]
                rows.append(dict(comparison='mira_full minus '+comparator,comparator=comparator,pattern=pattern,imputer=imputer,metric=metric,
                    difference=delta[j],conditional95_low=low,conditional95_high=high,
                    bonferroni98_333_low=adjusted[0],bonferroni98_333_high=adjusted[1],
                    primary_family_of_three=primary,bootstrap_draws=draws,bootstrap_seed=seed,
                    direction_favouring_full=('positive' if metric in ['accuracy','macro_f1'] else 'negative')))
    contrasts=pd.DataFrame(rows);assert len(contrasts)==84 and contrasts.primary_family_of_three.sum()==3
    contrasts.to_csv(out/'component-contrasts-all.csv',index=False)
    contrasts[contrasts.primary_family_of_three].to_csv(out/'primary-component-family-b50-linear-f1.csv',index=False)
    np.savez_compressed(out/'paired-bootstrap-statistics.npz',**stored)
    return contrasts


def fmt_mean_sd(row,metric,percent=False):
    scale=100 if percent else 1;digits=1 if percent else 3
    return f"${row[metric+'_mean']*scale:.{digits}f}\\pm{row[metric+'_std']*scale:.{digits}f}$"


def tex_escape(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def table(path,caption,label,headers,rows,column_spec=None):
    spec=column_spec or 'l'+'c'*(len(headers)-1)
    text='\\begin{table*}[t]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\\setlength{\\tabcolsep}{5pt}\\renewcommand{\\arraystretch}{1.25}\n'
    text+='\\begin{tabular}{'+spec+'}\n\\hline\n'+' & '.join(headers)+r'\\'+'\n\\hline\n'
    text+='\n'.join(' & '.join(row)+r'\\' for row in rows)+'\n\\hline\n\\end{tabular}\n\\end{table*}\n'
    path.write_text(text,encoding='utf-8')


def write_tables(summary,pf_summary,contrasts,out):
    folder=out/'tables';folder.mkdir(exist_ok=True)
    main=[]
    for v in VARIANTS:
        values=[]
        for pattern,imputer in [('complete','none'),('random','linear'),('contiguous','linear')]:
            r=summary[(summary.model==v)&(summary.pattern==pattern)&(summary.imputer==imputer)].iloc[0]
            values.append(fmt_mean_sd(r,'accuracy',True))
        main.append([LABELS[v],*values])
    table(folder/'held-accuracy-primary.tex','Held-domain accuracy (percent), pooling 1,260 distinct files within each training/mask cell. Mask metrics are averaged within training seed, followed by the mean and sample SD across five training seeds. R50 and B50 use linear repair. SD is conditional on fixed data and partitions, not uncertainty over new environments.','tab:frozen_accuracy',['Model','Complete','R50','B50'],main)
    family=[]
    for comparator in COMPARATORS:
        r=contrasts[contrasts.primary_family_of_three&(contrasts.comparator==comparator)].iloc[0]
        family.append(['Full minus '+LABELS[comparator],f"${r.difference:+.4f}$",f"$[{r.conditional95_low:+.4f},{r.conditional95_high:+.4f}]$",f"$[{r.bonferroni98_333_low:+.4f},{r.bonferroni98_333_high:+.4f}]$"])
    table(folder/'component-family-b50-linear.tex','Exploratory post-hoc component contrasts in macro F1 under 50\\% contiguous loss with linear repair. The point estimate averages the 25 training/mask-cell metrics. Ten thousand paired participant-cluster draws sample ten people with replacement within each of three fixed environments. All configurations share the same draws. Bonferroni ranges use 98.333\\% marginal coverage for this family of three; they remain conditional on fixed rooms, data, fitted models, calibration and masks. Positive differences favour full MIRA.','tab:frozen_component_family',['Contrast',r'$\Delta$ F1',r'Unadjusted 95\%',r'Bonferroni 98.333\%'],family)
    for pattern,imputer in GROUPS:
        body=[]
        for v in VARIANTS:
            r=summary[(summary.model==v)&(summary.pattern==pattern)&(summary.imputer==imputer)].iloc[0]
            body.append([LABELS[v],fmt_mean_sd(r,'accuracy',True),*[fmt_mean_sd(r,m) for m in ['macro_f1','nll','brier','ece']]])
        table(folder/f'held-{pattern}-{imputer}.tex',f'Held-domain results for {tex_escape(pattern)} input and {tex_escape(imputer)} repair. Means and sample SD use five training seeds after averaging five mask seeds where applicable. Accuracy is percent; the other metrics retain their original scales. No ensembling of probabilities is performed.','tab:frozen_'+pattern+'_'+imputer,['Model','Accuracy',r'Macro F1','NLL','Brier','ECE'],body)
    pf=[]
    for v in VARIANTS:
        for fold in [1,2,3]:
            values=[]
            for pattern,imputer in [('complete','none'),('random','linear'),('contiguous','linear')]:
                r=pf_summary[(pf_summary.model==v)&(pf_summary.held_environment==fold)&(pf_summary.pattern==pattern)&(pf_summary.imputer==imputer)].iloc[0]
                values.append(fmt_mean_sd(r,'accuracy',True))
            pf.append([LABELS[v],f'E{fold}',*values])
    table(folder/'per-environment-accuracy.tex','Held-domain accuracy by excluded environment and cohort. Each fold has 420 target files. Mask scores are averaged before the mean and sample SD over five training seeds. E1/E2/E3 are fixed domains, not independent samples of a room population.','tab:frozen_per_environment',['Model','Domain','Complete','R50 linear','B50 linear'],pf)
    (folder/'README.txt').write_text('Self-contained table fragments use standard LaTeX table/tabular commands; insert selected fragments in an already reviewed manuscript. The full CSVs are authoritative. Accuracy tables multiply fractional accuracy by100. F1 differences use the original0--1 scale. Family corrections apply only to the three declared B50-linear component F1 contrasts, not to all secondary results. No main.tex file was modified.\n')


def save_figure(fig,folder,name):
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight')
    plt.close(fig)


def figures(summary,pf_summary,contrasts,raw_confusions,conditions,classes,out):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    markers=['o','s','D','^','v','P'];positions=np.arange(3)
    primary=[('complete','none'),('random','linear'),('contiguous','linear')]
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.45))
    for ax,metric,title,scale in [(axes[0],'accuracy','(a) Accuracy (%)',100),(axes[1],'macro_f1','(b) Macro F1',1),(axes[2],'nll','(c) NLL',1)]:
        for j,v in enumerate(VARIANTS):
            rr=[summary[(summary.model==v)&(summary.pattern==p)&(summary.imputer==m)].iloc[0] for p,m in primary]
            ax.errorbar(positions+(j-2.5)*.04,[r[metric+'_mean']*scale for r in rr],yerr=[r[metric+'_std']*scale for r in rr],color=COLORS[j],marker=markers[j],capsize=1.5,lw=1.1,ms=3.8,label=LABELS[v])
        ax.set_xticks(positions,['C','R50','B50']);ax.set_title(title,fontsize=9);ax.grid(axis='y',alpha=.2)
        ax.tick_params(axis='both',labelsize=8)
        ax.set_xlim(-.28,2.28)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7.8,bbox_to_anchor=(.5,.005),frameon=False)
    fig.subplots_adjust(left=.075,right=.99,top=.91,bottom=.29,wspace=.33);save_figure(fig,folder,'held-model-comparison')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),sharey=True)
    for ax,pattern,title in zip(axes,['random','contiguous'],['Random 50% loss','Contiguous 50% loss']):
        for j,v in enumerate(VARIANTS):
            rr=[summary[(summary.model==v)&(summary.pattern==pattern)&(summary.imputer==m)].iloc[0] for m in ['linear','nearest','source_mean']]
            ax.errorbar(np.arange(3)+(j-2.5)*.035,[r.macro_f1_mean for r in rr],yerr=[r.macro_f1_std for r in rr],color=COLORS[j],marker=markers[j],capsize=2,ms=4,lw=1.1,label=LABELS[v])
        ax.set_xticks(np.arange(3),['Linear','Nearest','Source mean']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Held macro F1');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8.2,bbox_to_anchor=(.5,-.04));fig.tight_layout(rect=[0,.13,1,1]);save_figure(fig,folder,'held-repair-comparison')
    fam=contrasts[contrasts.primary_family_of_three].set_index('comparator').loc[COMPARATORS]
    fig,ax=plt.subplots(figsize=(7.2,3.0));yy=np.arange(3)
    for j,(_,r) in enumerate(fam.iterrows()):
        ax.plot([r.bonferroni98_333_low,r.bonferroni98_333_high],[j,j],color='#637786',lw=4,solid_capstyle='butt')
        ax.plot([r.conditional95_low,r.conditional95_high],[j,j],color='#00796B',lw=1.3)
        ax.plot([r.conditional95_low,r.conditional95_high],[j,j],linestyle='none',marker='|',color='#00796B',ms=9,mew=1.3)
        ax.plot(r.difference,j,'o',color='#162A35',ms=5)
    ax.axvline(0,color='#666666',ls='--',lw=.9);ax.set_yticks(yy,['Full minus '+LABELS[x] for x in COMPARATORS]);ax.invert_yaxis()
    ax.set_xlabel('Macro-F1 difference: positive favours full MIRA');ax.set_title('Post-hoc component family: B50 with linear repair')
    ax.legend([Line2D([0],[0],color='#00796B',lw=1.3,marker='|',ms=8),Line2D([0],[0],color='#637786',lw=4)],['Conditional 95%','Bonferroni 98.333% (three contrasts)'],fontsize=8,loc='best')
    ax.grid(axis='x',alpha=.15);fig.tight_layout();save_figure(fig,folder,'component-family-b50-linear')
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.25),sharey=True)
    for ax,fold in zip(axes,[1,2,3]):
        for j,v in enumerate(VARIANTS):
            rr=[pf_summary[(pf_summary.model==v)&(pf_summary.held_environment==fold)&(pf_summary.pattern==p)&(pf_summary.imputer==m)].iloc[0] for p,m in primary]
            ax.errorbar(positions+(j-2.5)*.04,[r.accuracy_mean*100 for r in rr],yerr=[r.accuracy_std*100 for r in rr],color=COLORS[j],marker=markers[j],ms=3,capsize=1.5,lw=1,label=LABELS[v])
        ax.set_xticks(positions,['Clean','R50','B50']);ax.set_title(f'Excluded E{fold} + cohort');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Accuracy (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.04));fig.tight_layout(rect=[0,.15,1,1]);save_figure(fig,folder,'accuracy-per-environment')
    short=['Sit','Fall','Lie','Stand','Walk','SitDn','Pick']
    for v in VARIANTS:
        fig,axes=plt.subplots(2,3,figsize=(7.2,5.2));last=None
        for row,(pattern,imputer,title) in enumerate([('complete','none','Complete'),('contiguous','linear','B50 linear')]):
            ids=group_ids(conditions,pattern,imputer)
            for col,fold in enumerate([1,2,3]):
                c=raw_confusions[v][col][:,ids].mean((0,1));den=c.sum(1,keepdims=True);rate=np.divide(c,den,out=np.zeros_like(c),where=den>0)
                ax=axes[row,col];last=ax.imshow(rate,vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
                for a in range(7):
                    for b in range(7):ax.text(b,a,f'{100*rate[a,b]:.0f}',ha='center',va='center',fontsize=7.8,color='white' if rate[a,b]>.52 else '#18232E')
                ax.set_xticks(np.arange(7),short,rotation=45,ha='right',fontsize=7.6);ax.set_yticks(np.arange(7),short,fontsize=7.6);ax.set_title(f'E{fold}: {title}',fontsize=9)
                if col==0:ax.set_ylabel('True class')
                if row==1:ax.set_xlabel('Predicted class')
        fig.suptitle(LABELS[v]+': mean conditional class fractions (%)',fontsize=10)
        fig.tight_layout(rect=[0,0,.94,.96]);bar=fig.add_axes([.945,.20,.015,.60]);fig.colorbar(last,cax=bar).set_label('Within-true-class fraction',fontsize=8)
        save_figure(fig,folder,'confusions-'+v)
    (folder/'FIGURE-CAPTIONS.txt').write_text('Model comparison: held accuracy, macro F1 and source-temperature-calibrated NLL. C denotes complete input; R50 and B50 denote random and contiguous 50% loss with linear repair. Higher accuracy/F1 and lower NLL are preferred. Each metric pools 1,260 held files per training/mask cell. Model/repair/per-environment plots show means of five training-seed summaries after averaging the five mask seeds per corrupted case; error bars are sample SD over those five training seeds, conditional on the fixed recordings/partitions. They are not confidence intervals for new environments.\nComponent forest: exploratory post-hoc participant-bootstrap percentile ranges, using 10,000 common draws within three fixed environment/cohort strata. Thin green segments with terminal ticks mark unadjusted 95% ranges; thick grey segments mark Bonferroni 98.333% ranges for only the three B50-linear F1 contrasts. No new-model fitting or threshold selection.\nConfusions: each matrix averages the row-normalized confusion fractions over fitted seed/mask cells within the specified excluded environment/cohort. The percentage annotations are conditional fractions, not independent trial counts. Complete input has five fitted cells; B50 linear has 25.\nClass order: '+json.dumps(classes)+'\n')


def self_check():
    from sklearn.metrics import accuracy_score,f1_score,log_loss
    rng=np.random.default_rng(9);y=np.tile(np.arange(7),11);p=rng.random((len(y),7));p/=p.sum(1,keepdims=True)
    actual=score(y,p)
    assert abs(actual['accuracy']-accuracy_score(y,p.argmax(1)))<1e-14
    assert abs(actual['macro_f1']-f1_score(y,p.argmax(1),labels=range(7),average='macro'))<1e-14
    assert abs(actual['nll']-log_loss(y,p,labels=range(7)))<1e-14
    cm=np.bincount(y*7+p.argmax(1),minlength=49).reshape(7,7).astype(float)
    np.testing.assert_allclose(cm_scores(cm),[actual['accuracy'],actual['macro_f1']],atol=1e-14)
    a=rng.normal(size=10000);u=np.percentile(a,[2.5,97.5]);b=np.percentile(a,[100*.05/(2*3),100*(1-.05/(2*3))])
    assert b[0]<=u[0]<=u[1]<=b[1]
    print('Independent scoring and Bonferroni-quantile checks passed on explicitly synthetic helper inputs; no study data read.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-archive',type=Path);ap.add_argument('--evaluation-archive',type=Path)
    ap.add_argument('--data-root',type=Path,default=WORKSPACE/'mira-csi/data')
    ap.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    ap.add_argument('--output-root',type=Path,default=WORKSPACE/'output/peer-review-2026-10-28/frozen-evaluation-analysis')
    ap.add_argument('--bootstrap',type=int,default=10000);ap.add_argument('--seed',type=int,default=20261028)
    ap.add_argument('--bootstrap-indices',type=Path);ap.add_argument('--self-check',action='store_true')
    args=ap.parse_args()
    if args.self_check:self_check();return
    if not args.source_archive or not args.evaluation_archive:ap.error('Both immutable source and separate evaluation archives are required.')
    assert args.source_archive.resolve()!=args.evaluation_archive.resolve()
    assert args.bootstrap==10000 and args.seed==20261028,'This analysis uses the declared10000 draws and seed20261028; alternative analyses need an explicit separate specification.'
    assert sha(args.protocol)==PROTOCOL_SHA
    config=json.loads(args.protocol.read_text())
    for name,digest in config['input_sha256'].items():assert sha(args.data_root/name)==digest
    for name,digest in config['implementation_sha256'].items():assert sha(HERE/name)==digest
    entries=json.loads((args.data_root/'manifest.json').read_text())['selected'];assert len(entries)==1260
    out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    source_before=sha(args.source_archive);eval_before=sha(args.evaluation_archive);start=time.perf_counter()
    source=Archive(args.source_archive);evaluation=Archive(args.evaluation_archive)
    try:
        pools,truth,conditions,audit=validate_and_pool(source,evaluation,config,entries,out)
        summary,pf_summary,participant,people,strata,confusions=summarise(pools,truth,conditions,entries,out)
        contrasts=bootstrap_analysis(participant,conditions,people,strata,out,args.bootstrap,args.seed,args.bootstrap_indices)
        # Bootstrap point estimates must equal the reported full-grid means.
        for _,r in contrasts.iterrows():
            l=summary[(summary.model=='mira_full')&(summary.pattern==r['pattern'])&(summary.imputer==r.imputer)].iloc[0]
            rr=summary[(summary.model==r.comparator)&(summary.pattern==r['pattern'])&(summary.imputer==r.imputer)].iloc[0]
            assert abs((l[r.metric+'_mean']-rr[r.metric+'_mean'])-r.difference)<1e-12
        write_tables(summary,pf_summary,contrasts,out);figures(summary,pf_summary,contrasts,confusions,conditions,config['classes'],out)
    finally:source.close();evaluation.close()
    assert sha(args.source_archive)==source_before and sha(args.evaluation_archive)==eval_before
    audit.update(passed=True,protocol_sha256=PROTOCOL_SHA,source_archive_sha256=source_before,evaluation_archive_sha256=eval_before,
        archives_unchanged_after_analysis=True,pooled_held_cell_rows=930,mask_averaged_seed_rows=210,summary_rows=42,
        bootstrap_draws=args.bootstrap,bootstrap_seed=args.seed,primary_family_size=3,
        primary_family='Full MIRA minus each component-removal configuration, B50 linear repair, macro F1 only',
        family_interval_coverage=1-.05/3,unadjusted_interval_coverage=.95,
        estimator='Score pooled1260 held files within each training/mask cell; mean over masks within training seed; mean and sample SD across five training seeds. Bootstrap averages the same cell metrics, not ensemble probabilities.',
        inference_scope='Post-hoc exploratory and conditional on fixed environments/cohorts, recordings, models, calibration and mask grid. No uncertainty over new rooms or retraining; no preregistration, novelty or deployment claim. Secondary intervals are unadjusted; only the three named F1 contrasts receive Bonferroni correction.',
        elapsed_seconds=time.perf_counter()-start,analysis_source_sha256=sha(__file__))
    write_json(out/'independent-analysis-receipt.json',audit)
    (out/'README.md').write_text('# Frozen-evaluation analysis\n\nAll archives were read without extraction or modification. No model or calibration was refitted.\n\nThe original source ZIP remains the immutable training freeze; the separate evaluation ZIP supplies only the declared frozen evaluation. The receipt distinguishes actual checkpoint-byte verification from a packaged post-evaluation hash manifest.\n\nThe primary component family contains exactly three exploratory post-hoc macro-F1 contrasts under B50linear. Conditional95% and Bonferroni98.333% ranges share the same10000 participant-cluster draws. Other contrasts are secondary and unadjusted. These intervals do not establish a causal room effect, model novelty or deployment readiness.\n\nAccuracy CSV values are fractions; LaTeX accuracy tables use percent. Macro-F1 differences remain on the0--1 scale. Neither seeds, corruptions, recordings nor bootstrap draws are additional recruited participants.\n\nNo manuscript or training source was modified, and nothing was published.\n',encoding='utf-8')
    print('VERIFIED ANALYSIS COMPLETE',json.dumps({k:audit[k] for k in ['fits','metric_rows','probability_vectors','summary_rows','bootstrap_draws','archives_unchanged_after_analysis','elapsed_seconds']}),flush=True)
    print(contrasts[contrasts.primary_family_of_three].to_string(index=False),flush=True)


if __name__=='__main__':main()
