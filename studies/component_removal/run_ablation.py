"""Frozen-protocol, resumable source-only training and explicit held evaluation.

Default mode never infers or scores a target recording. --smoke is a separately
labelled one-epoch execution check, not an experimental result.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,random,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score,log_loss
from models import build_model
from operators import make_mask,repair,gap_distance

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'protocol.json').read_text())
HP=CONFIG['training']
CONDITIONS=[('complete',0.,0,'none')]+[(p,.5,s,m)
    for p in ['random','contiguous'] for m in CONFIG['imputers'] for s in CONFIG['mask_seeds']]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def rows(p,records):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def atomic_save(obj,path):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);os.replace(temp,path)
def prob(logits,T=1.):
    z=np.asarray(logits,np.float64)/T;z-=z.max(1,keepdims=True)
    e=np.exp(z);return e/e.sum(1,keepdims=True)
def scoring(y,p):
    pred=p.argmax(1);correct=pred==y;confidence=p.max(1);ece=0.
    for b in range(10):
        k=(confidence>=b/10)&((confidence<(b+1)/10) if b<9 else confidence<=1)
        if k.any():ece+=k.mean()*abs(confidence[k].mean()-correct[k].mean())
    return dict(n=len(y),correct=int(correct.sum()),accuracy=float(correct.mean()),
        macro_f1=float(f1_score(y,pred,labels=range(7),average='macro',zero_division=0)),
        nll=float(log_loss(y,p,labels=range(7))),
        brier=float(np.square(p-np.eye(7)[y]).sum(1).mean()),ece=float(ece))

@torch.inference_mode()
def infer(model,x,m,d,mean,scale,device):
    model.eval();out=[]
    for i in range(0,len(x),128):
        xx=torch.as_tensor((x[i:i+128]-mean)/scale,device=device)
        mm=torch.as_tensor(m[i:i+128],device=device);dd=torch.as_tensor(d[i:i+128],device=device)
        out.append(model(xx,mm,dd).cpu().numpy())
    return np.concatenate(out)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def train(X,y,entries,tr,va,mean,scale,variant,seed,fold,device,out,smoke):
    seed_all(seed);model=build_model(variant).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=HP['learning_rate'],weight_decay=HP['weight_decay'])
    rng=np.random.default_rng(seed);tag=f'{variant}_E{fold}_s{seed}'
    latest=out/'checkpoints'/f'{tag}.latest.pt';final=out/'checkpoints'/f'{tag}.pt'
    finished=out/'runs'/f'{tag}.training.json';protocol_hash=sha(HERE/'protocol.json')
    if finished.exists():
        meta=json.loads(finished.read_text());assert meta['protocol_sha256']==protocol_hash
        assert meta['checkpoint_sha256']==sha(final)
        state=torch.load(final,map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict']);return model,meta
    best=float('inf');stale=0;history=[];best_state=None;best_epoch=None;first=0
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==protocol_hash
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for vv in opt.state.values():
            for k,v in vv.items():
                if isinstance(v,torch.Tensor):vv[k]=v.to(device)
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        best=state['best'];stale=state['stale'];history=state['history'];best_state=state['best_state']
        best_epoch=state['best_epoch'];first=state['next_epoch']
    start=time.perf_counter();epochs=1 if smoke else HP['max_epochs']
    if stale>=HP['patience']:first=epochs
    for epoch in range(first,epochs):
        model.train();order=rng.permutation(len(tr));loss_sum=0.;n=0;ep_start=time.perf_counter()
        for begin in range(0,len(order),HP['batch_size']):
            batch=order[begin:begin+HP['batch_size']];ix=tr[batch]
            xx=X[ix].copy();mm=np.ones((len(ix),1,500),np.float32);dd=np.zeros_like(mm)
            for j,idx in enumerate(ix):
                if variant!='resnet1d_clean' and rng.random()<.5:
                    pattern=str(rng.choice(['random','contiguous']));fraction=float(rng.choice([.1,.3,.5]))
                    method=str(rng.choice(CONFIG['imputers']));aug_seed=int(rng.integers(0,2**31-1))
                    observed=make_mask(entries[idx]['name'],pattern,fraction,aug_seed)
                    xx[j]=repair(xx[j],observed,method,mean.ravel());mm[j,0]=observed;dd[j,0]=gap_distance(observed)
            bx=torch.as_tensor((xx-mean)/scale,device=device);bm=torch.as_tensor(mm,device=device)
            bd=torch.as_tensor(dd,device=device);by=torch.as_tensor(y[ix],dtype=torch.long,device=device)
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(bx,bm,bd),by)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite source-training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            loss_sum+=float(loss.detach())*len(ix);n+=len(ix)
        vm=np.ones((len(va),1,500),np.float32);vd=np.zeros_like(vm)
        vp=prob(infer(model,X[va],vm,vd,mean,scale,device));nll=float(log_loss(y[va],vp,labels=range(7)))
        record=dict(epoch=epoch+1,train_nll=loss_sum/n,source_validation_nll=nll,
            source_validation_accuracy=float((vp.argmax(1)==y[va]).mean()),epoch_seconds=time.perf_counter()-ep_start)
        history.append(record)
        if nll<best-HP['min_delta']:
            best=nll;stale=0;best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        atomic_save(dict(protocol_sha256=protocol_hash,model=model.state_dict(),optimizer=opt.state_dict(),
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            best=best,stale=stale,history=history,best_state=best_state,best_epoch=best_epoch,
            next_epoch=epoch+1),latest)
        print(json.dumps({'tag':tag,'mode':'SMOKE_NOT_RESEARCH' if smoke else 'source_training',**record}),flush=True)
        if stale>=HP['patience']:break
    assert best_state is not None;model.load_state_dict(best_state)
    vm=np.ones((len(va),1,500),np.float32);vd=np.zeros_like(vm)
    logits=infer(model,X[va],vm,vd,mean,scale,device)
    optimum=minimize_scalar(lambda t:log_loss(y[va],prob(logits,t),labels=range(7)),bounds=(.1,10),method='bounded')
    assert optimum.success
    atomic_save(dict(state_dict=best_state,mean=mean,scale=scale,variant=variant,seed=seed,
        target_environment=fold,best_epoch=best_epoch,classes=CONFIG['classes'],temperature=float(optimum.x),
        protocol_sha256=protocol_hash,smoke=smoke),final)
    np.savez_compressed(out/'validation'/f'{tag}.npz',clean_logits=logits,truth=y[va],indices=va)
    rows(out/'histories'/f'{tag}.csv',history)
    meta=dict(mode='SMOKE_NOT_RESEARCH' if smoke else 'completed_source_training',target_scored=False,
        variant=variant,training_seed=seed,held_environment=fold,epochs=len(history),best_epoch=best_epoch,
        trainable_parameters=sum(p.numel() for p in model.parameters()),temperature=float(optimum.x),
        source_validation_nll=float(optimum.fun),training_seconds=time.perf_counter()-start,
        protocol_sha256=protocol_hash,checkpoint_sha256=sha(final),source_train_files=len(tr),source_validation_files=len(va))
    dump(finished,meta);return model,meta

def evaluate(model,meta,X,y,entries,indices,mean,scale,device,out):
    assert not meta['mode'].startswith('SMOKE');predictions=[];records=[]
    held=np.array([entries[i]['environment']==meta['held_environment'] for i in indices])
    for cid,(pattern,fraction,mask_seed,method) in enumerate(CONDITIONS):
        xx=[];mm=[];dd=[]
        for i in indices:
            m=make_mask(entries[i]['name'],pattern,fraction,mask_seed)
            xx.append(repair(X[i],m,method,mean.ravel()) if fraction else X[i]);mm.append(m.astype(np.float32)[None]);dd.append(gap_distance(m)[None])
        logits=infer(model,np.array(xx,np.float32),np.array(mm),np.array(dd),mean,scale,device)
        p=prob(logits,meta['temperature']);assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1.,atol=1e-12)
        predictions.append(p)
        for domain,k in [('held',held),('familiar',~held)]:
            records.append(dict(model=meta['variant'],training_seed=meta['training_seed'],held_environment=meta['held_environment'],
                domain=domain,condition_id=cid,pattern=pattern,missing_fraction=fraction,mask_seed=mask_seed,imputer=method,**scoring(y[indices][k],p[k])))
    tag=f"{meta['variant']}_E{meta['held_environment']}_s{meta['training_seed']}"
    np.savez_compressed(out/'predictions'/f'{tag}.npz',probabilities=np.array(predictions),indices=indices,truth=y[indices],held=held,condition_ids=np.arange(31))
    rows(out/'metrics'/f'{tag}.csv',records)
    dump(out/'runs'/f'{tag}.evaluation.json',dict(status='frozen evaluation completed',target_scored=True,protocol_sha256=meta['protocol_sha256'],prediction_sha256=sha(out/'predictions'/f'{tag}.npz')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=HERE.parents[1]/'mira-csi/data')
    ap.add_argument('--output-root',type=Path,default=HERE/'results');ap.add_argument('--variants',nargs='+',default=CONFIG['primary_variants'])
    ap.add_argument('--seeds',nargs='+',type=int,default=CONFIG['training_seeds']);ap.add_argument('--folds',nargs='+',type=int,default=CONFIG['folds'])
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--evaluate',action='store_true');args=ap.parse_args()
    if args.smoke and args.evaluate:ap.error('Smoke mode cannot score target/familiar tests.')
    for name,value in CONFIG['implementation_sha256'].items():assert sha(HERE/name)==value,name
    assert set(args.variants)<=set(CONFIG['variants']) and set(args.seeds)<=set(CONFIG['training_seeds']) and set(args.folds)<=set(CONFIG['folds'])
    out=args.output_root/('smoke' if args.smoke else 'study')
    for name in ['checkpoints','histories','validation','runs','metrics','predictions']:(out/name).mkdir(parents=True,exist_ok=True)
    for name,value in CONFIG['input_sha256'].items():assert sha(args.data_root/name)==value,name
    X=np.load(args.data_root/'band_streams.npy');entries=json.loads((args.data_root/'manifest.json').read_text())['selected']
    assert X.shape==(1260,18,500) and np.isfinite(X).all()
    y=np.array([CONFIG['classes'].index(e['activity']) for e in entries]);env=np.array([e['environment'] for e in entries]);role=np.array([e['role'] for e in entries]);subjects=np.array([e['subject'] for e in entries])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    dump(out/'runtime.json',dict(torch=torch.__version__,numpy=np.__version__,device=str(device),device_name=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',protocol_sha256=sha(HERE/'protocol.json'),target_evaluation_requested=args.evaluate))
    dump(out/'conditions.json',CONDITIONS);audit=[]
    for fold in args.folds:
        tr=np.flatnonzero((env!=fold)&(role=='train'));va=np.flatnonzero((env!=fold)&(role=='validation'))
        familiar=np.flatnonzero((env!=fold)&(role=='familiar_test'));held=np.flatnonzero(env==fold)
        assert [len(x) for x in [tr,va,familiar,held]]==[504,168,168,420]
        groups=[set(subjects[x]) for x in [tr,va,familiar,held]]
        assert all(not groups[a]&groups[b] for a in range(4) for b in range(a+1,4))
        audit.append(dict(target=fold,participants=[sorted(int(s) for s in g) for g in groups],counts=[504,168,168,420]))
        mean=X[tr].mean((0,2),keepdims=True).astype(np.float32);scale=np.maximum(X[tr].std((0,2),keepdims=True),1e-6).astype(np.float32)
        for seed in args.seeds:
            for variant in args.variants:
                model,meta=train(X,y,entries,tr,va,mean,scale,variant,seed,fold,device,out,args.smoke)
                if args.evaluate:evaluate(model,meta,X,y,entries,np.r_[familiar,held],mean,scale,device,out)
    dump(out/'split_audit.json',audit)
    print('SMOKE ONLY, NO TARGET SCORED.' if args.smoke else 'REQUESTED SOURCE TRAINING FINISHED; target scoring is opt-in with --evaluate.',flush=True)

if __name__=='__main__':main()
