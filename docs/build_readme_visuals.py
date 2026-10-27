"""Rebuild repository visuals from the released public CSI example.

No RF field, human silhouette or model prediction is synthesized.
"""
from pathlib import Path
import io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets';ASSETS.mkdir(parents=True,exist_ok=True)
recording=np.load(ROOT/'results/demo_input.npy')
observed=np.load(ROOT/'results/demo_observed.npy').astype(bool)
time=np.arange(500)
repaired=np.asarray([np.interp(time,time[observed],row[observed]) for row in recording])
lo,hi=np.quantile(recording,[.02,.98])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

fig=plt.figure(figsize=(12,3.9),facecolor='#091522')
fig.text(.045,.76,'MIRA-CSI',fontsize=49,color='#f3f7fb',weight='bold')
fig.text(.047,.61,'Missingness-informed Wi-Fi activity recognition',fontsize=14,color='#69d6cf')
fig.text(.047,.48,'Developed by Eng. Abdessamad Bourkibate',fontsize=11,color='#c2cedb')
fig.text(.047,.39,'Recorded CSI  /  reproducible models  /  transparent evaluation',fontsize=9.8,color='#91a9bf')
ax=fig.add_axes([.67,.38,.29,.44]);ax.imshow(recording,aspect='auto',cmap='viridis',vmin=lo,vmax=hi)
ax.set_xticks([]);ax.set_yticks([])
for spine in ax.spines.values():spine.set_color('#355367')
ax.set_title('REAL CSI AMPLITUDE · PUBLIC RECORDING',color='#91a9bf',fontsize=7.5,pad=9)
metrics=[('1,260','recordings'),('30','participants'),('3','environments'),('5','training seeds')]
for i,(value,label) in enumerate(metrics):
    x=.047+i*.232
    fig.patches.append(FancyBboxPatch((x,.065),.21,.20,transform=fig.transFigure,
                         boxstyle='round,pad=.009,rounding_size=.012',facecolor='#112638',edgecolor='#24475a',lw=.9))
    fig.text(x+.018,.17,value,fontsize=21,color='#edf7fa',weight='bold')
    fig.text(x+.018,.105,label.upper(),fontsize=8,color='#8fb1c3')
fig.savefig(ASSETS/'mira-csi-banner.png',dpi=160,facecolor=fig.get_facecolor(),bbox_inches='tight',pad_inches=.12)
plt.close(fig)

images=[];durations=[]
stages=[('01  RECORDED SIGNAL',recording,'All 500 normalized packet positions available'),
        ('02  SYNTHETIC BLOCK LOSS',np.where(observed[None],recording,np.nan),'250 positions removed jointly across 18 bands'),
        ('03  LINEAR REPAIR',repaired,'Only observed boundaries determine the filled interval')]
for i,(title,values,note) in enumerate(stages):
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),gridspec_kw={'width_ratios':[1.18,1]})
    fig.patch.set_facecolor('#f5f8fb')
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('#d6e0e8')
    axes[0].imshow(values,aspect='auto',cmap=cmap,vmin=lo,vmax=hi)
    axes[0].set_ylabel('Amplitude band');axes[0].set_xlabel('Normalized packet position')
    axes[0].set_title('18 CSI amplitude bands',fontsize=10)
    axes[1].plot(time,recording[8],color='#8b9bab',lw=1,label='Original recorded band')
    if i==1:
        axes[1].plot(time,np.where(observed,recording[8],np.nan),color='#246b8a',lw=1.7,label='Retained observations')
    elif i==2:
        axes[1].plot(time,repaired[8],color='#be7b22',lw=1.5,label='Linear repair')
    else:
        axes[1].plot(time,recording[8],color='#246b8a',lw=1.4)
    if i:
        missing=np.flatnonzero(~observed)
        axes[1].axvspan(missing[0],missing[-1],color='#efc56a',alpha=.20)
    axes[1].set_xlabel('Normalized packet position');axes[1].set_ylabel('Scaled amplitude (dB)')
    axes[1].set_title('One band, same recording',fontsize=10);axes[1].legend(fontsize=7,loc='lower left')
    axes[1].grid(alpha=.16)
    for a in axes:
        a.spines[['top','right']].set_visible(False);a.tick_params(labelsize=8)
    fig.suptitle(title,color='#15374a',fontsize=13,weight='bold',y=.97)
    fig.text(.5,.025,note+' · Offline demonstration; animation time is not acquisition time.',ha='center',fontsize=8,color='#526777')
    fig.tight_layout(rect=(0,.075,1,.90))
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110);plt.close(fig);buffer.seek(0)
    images.append(Image.open(buffer).convert('RGB'));durations.append(1800 if i<2 else 2400)
images[0].save(ASSETS/'csi-repair-demo.gif',save_all=True,append_images=images[1:],duration=durations,loop=0,optimize=False)
(ASSETS/'VISUAL-PROVENANCE.json').write_text(json.dumps(dict(
    source='First E3 public recording in the evaluated manifest; same retained inference example',
    original_shape=[18,500],mask='Fixed contiguous 50%, seed 20261026',
    display_limits='2nd/98th percentiles of this example, shared across all frames; display clipping only',
    color='Scaled CSI amplitude, not body temperature or RF-field intensity',
    transformations='Measured array, exact saved missing mask, observed-only linear interpolation',
    data_license='CC BY 4.0, Alsaify et al., DOI 10.17632/v38wjmz6f6.2'),indent=2))
print('Saved data-derived banner and three-stage CSI repair GIF.')
