"""Rebuild the fixed real-CSI tensor from pinned, hash-verified public files.

Dataset/scaling source: Alsaify et al., DOI 10.1016/j.dib.2020.106534,
and the author repository recorded in data/manifest.json (original MIT code).
Radio scaling follows the author's get_scaled_csi.m/Halperin convention.
"""
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
import hashlib,json
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'data/manifest.json').read_text())
base='https://raw.githubusercontent.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments/'

def convert(entry):
    path=ROOT/'data/raw'/f"E{entry['environment']}"/entry['name']
    if path.exists():
        raw=path.read_bytes()
    else:
        url=base+manifest['commit']+f"/MAT/E{entry['environment']}/"+entry['name']
        raw=urlopen(Request(url,headers={'User-Agent':'MIRA-CSI-reproducible-research'}),timeout=90).read()
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    assert hashlib.sha256(raw).hexdigest()==entry['sha256'],entry['name']
    packets=loadmat(path,simplify_cells=True)['data']
    packets=packets if isinstance(packets,list) else list(np.atleast_1d(packets))
    h=np.stack([np.asarray(p['csi']) for p in packets])
    assert h.ndim==3 and h.shape[1:]==(3,30)
    assert all(p['Nrx']==3 and p['Ntx']==1 for p in packets)
    power=(np.abs(h)**2).sum((1,2))
    rssi=np.array([[p[k] for k in ['rssi_a','rssi_b','rssi_c']] for p in packets],dtype=float)
    magnitude=np.where(rssi!=0,10**(rssi/10),0).sum(1)
    agc=np.array([p['agc'] for p in packets]);noise=np.array([p['noise'] for p in packets])
    noise=np.where(noise==-127,-92,noise)
    valid=(power>0)&(magnitude>0)&np.isfinite(h).all((1,2))
    assert valid.sum()>=2
    scale=np.zeros(len(packets))
    scale[valid]=magnitude[valid]*10**((-44-agc[valid])/10)/(power[valid]/30)
    total_noise=10**(noise/10)+scale*3
    amplitude=np.abs(h[valid])*np.sqrt(scale[valid]/total_noise[valid])[:,None,None]
    db=20*np.log10(np.maximum(amplitude,1e-12))
    bands=db.reshape(len(db),3,6,5).mean(3).reshape(len(db),18).T
    position=np.flatnonzero(valid);normalized=np.linspace(position[0],position[-1],500)
    result=np.array([np.interp(normalized,position,row) for row in bands],dtype=np.float32)
    assert result.shape==(18,500) and np.isfinite(result).all()
    return result

def main():
    result=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        # map retains manifest order, even if network fetches finish out of order.
        for i,stream in enumerate(pool.map(convert,manifest['selected']),1):
            result.append(stream)
            if i%100==0:print('Hash-verified and processed:',i,'/1260',flush=True)
    path=ROOT/'data/band_streams.npy';np.save(path,np.asarray(result,dtype=np.float32))
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    expected=json.loads((ROOT/'input_receipt.json').read_text())['sha256']['band_streams.npy']
    assert actual==expected, 'Reconstructed tensor differs from the evaluated input; do not claim identical reproduction.'
    print('PASS: rebuilt exact evaluated tensor SHA256',actual)

if __name__=='__main__':main()
