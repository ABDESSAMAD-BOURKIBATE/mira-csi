"""Render transparent README visuals; keep measured values and paper figures intact.

Run after downloading the released results into results/. Requires Pillow,
Matplotlib, NumPy and pandas. The three source GIFs are conceptual artwork;
their foreground is recolored after removing the cream video background.
"""
from pathlib import Path
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets'
THEMES={
    'light':dict(text='#18344c',line='#2c7281',amber='#aa6818',grid='#688091',node='#e7eff6',
                 colors=['#456a9c','#ba701e','#007e7b','#805ba0']),
    'dark':dict(text='#dbe5ee',line='#6dd9dc',amber='#f3c16c',grid='#8294a8',node='#1c2c3c',
                colors=['#84b1ed','#efb263','#65d5cf','#c5a4e6']),
}

def style(theme):
    c=THEMES[theme]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'text.color':c['text'],'axes.labelcolor':c['text'],'axes.edgecolor':c['grid'],
        'xtick.color':c['text'],'ytick.color':c['text'],'axes.titlecolor':c['text'],
        'axes.facecolor':'none','figure.facecolor':'none','savefig.transparent':True,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    return c

def conceptual_clips(theme):
    c=THEMES[theme]
    tint=np.array([int(c['line'][i:i+2],16) for i in (1,3,5)],dtype=np.uint8)
    for case in ['complete','random-loss','block-loss']:
        source=Image.open(OUT/f'case-{case}.gif');frames=[];durations=[]
        for frame in ImageSequence.Iterator(source):
            rgb=np.asarray(frame.convert('RGB'),dtype=np.float32)
            # Cream background has low blue/green chroma. Retain teal strokes
            # and the dark data-source symbol; no experimental values involved.
            chroma=np.maximum((rgb[:,:,1]-rgb[:,:,0]-4)/25,
                              (rgb[:,:,2]-rgb[:,:,0]-2)/25)
            dark=(115-rgb.max(axis=2))/70
            alpha=(np.clip(np.maximum(chroma,dark),0,1)*255).astype(np.uint8)
            rgba=np.empty((*alpha.shape,4),dtype=np.uint8);rgba[:,:,:3]=tint;rgba[:,:,3]=alpha
            frames.append(Image.fromarray(rgba));durations.append(frame.info.get('duration',125))
        frames[0].save(OUT/f'case-{case}-{theme}.webp',save_all=True,append_images=frames[1:],
                       duration=durations,loop=0,lossless=True,method=6)

def repair_clip(theme):
    c=style(theme)
    x=np.load(ROOT/'results/demo_input.npy');obs=np.load(ROOT/'results/demo_observed.npy').astype(bool)
    t=np.arange(x.shape[-1]);gap=np.flatnonzero(~obs)
    repair=np.interp(t,t[obs],x[8,obs]);frames=[]
    for filled in [False,True]:
        fig,ax=plt.subplots(figsize=(8,4.5),dpi=80)
        ax.plot(t,x[8],color=c['grid'],alpha=.65,lw=1,label='Original measured band')
        ax.plot(t,np.where(obs,x[8],np.nan),color=c['line'],lw=1.5,label='Retained observations')
        if filled:ax.plot(t[gap],repair[gap],color=c['amber'],lw=2,ls='--',label='Observed-only linear repair')
        ax.axvspan(gap[0],gap[-1],color=c['amber'],alpha=.09)
        ax.set_xlabel('Normalized packet position');ax.set_ylabel('Scaled CSI amplitude (dB)')
        ax.set_title('Real CSI example: contiguous loss and linear repair',loc='left',fontsize=11,pad=14)
        ax.grid(alpha=.15,color=c['grid']);ax.legend(loc='lower left',fontsize=8)
        fig.tight_layout(pad=2)
        buf=io.BytesIO();fig.savefig(buf,format='png',transparent=True);plt.close(fig);buf.seek(0)
        frames.append(Image.open(buf).convert('RGBA'))
    frames[0].save(OUT/f'case-linear-repair-{theme}.webp',save_all=True,
                  append_images=frames[1:],duration=[1400,2200],loop=0,lossless=True,method=6)

def architecture(theme):
    c=style(theme);fig,ax=plt.subplots(figsize=(11.5,3.8));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.08',
                                   facecolor=c['node'],edgecolor='none'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9.4,linespacing=1.5)
    def arrow(a,b,color=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.3,color=color or c['line']))
    box(.12,1.35,2.1,1.2,'18 amplitude bands\nTrain-only normalization\n500 packet positions')
    box(2.8,1.35,1.7,1.2,'Concatenate\n18 + mask + gap\n20 input channels')
    box(5.04,1.35,2.1,1.2,'Temporal CNN\n32 → 64 → 128\nk = 7, 5, 3; stride 2')
    box(7.7,1.35,1.75,1.2,'Coverage-weighted\ntemporal pooling\n128 features')
    box(10,1.35,1.75,1.2,'Dropout + linear\n7 activity logits\nScalar calibration')
    for a,b in [((2.3,1.95),(2.68,1.95)),((4.58,1.95),(4.92,1.95)),((7.22,1.95),(7.58,1.95)),((9.53,1.95),(9.88,1.95))]:arrow(a,b)
    box(2.8,2.98,2.1,.62,'Observed mask m; gap d')
    for a,b in [((3.65,2.9),(3.65,2.65)),((4.98,3.29),(8.57,3.29)),((8.57,3.29),(8.57,2.65))]:arrow(a,b,c['amber'])
    ax.text(6.7,3.52,'Mask → local coverage weights',ha='center',fontsize=9,color=c['amber'])
    ax.text(6,.79,'GroupNorm and GELU follow each convolution. Coverage is a deterministic proxy, not calibrated confidence.',ha='center',fontsize=9)
    ax.text(6,.28,'CNN comparators: clean training or source-only loss augmentation. MIRA-CSI: 40,871 parameters.',ha='center',fontsize=9)
    fig.tight_layout(pad=.5);fig.savefig(OUT/f'architecture-{theme}.png',dpi=160,bbox_inches='tight',transparent=True);plt.close(fig)

def comparison(theme):
    c=style(theme);data=pd.read_csv(ROOT/'evaluation/summary.csv')
    models=['cnn_clean','cnn_augmented','mira_csi','minirocket_9996']
    labels=['CNN','CNN + loss augmentation','MIRA-CSI','MiniRocket (9,996)']
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.1),sharey=True)
    for ax,domain,title in zip(axes,['familiar','held'],['Familiar source environments','Excluded environment + cohort']):
        for j,(model,label,color) in enumerate(zip(models,labels,c['colors'])):
            sub=data[(data.model==model)&(data.domain==domain)];means=[];sds=[]
            for pattern,imputer in [('complete','none'),('random','linear'),('contiguous','linear')]:
                row=sub[(sub.pattern==pattern)&(sub.imputer==imputer)];assert len(row)==1
                means.append(row.iloc[0].accuracy_mean*100);sds.append(row.iloc[0].accuracy_std*100)
            ax.errorbar(np.arange(3)+(j-1.5)*.055,means,yerr=sds,label=label,color=color,
                        marker=['o','s','D','^'][j],capsize=3,lw=1.5)
        ax.set_xticks(range(3),['Complete','Random 50%','Block 50%']);ax.set_title(title)
        ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2,color=c['grid'])
    axes[0].set_ylabel('Accuracy (%)');axes[1].legend(fontsize=8,loc='upper right')
    fig.text(.5,.01,'Mean ± sample SD of five paired seeds. Each seed pools three folds; linear repair.\nFamiliar files recur across folds.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(OUT/f'comparison-{theme}.png',dpi=160,transparent=True);plt.close(fig)

if __name__=='__main__':
    for theme in THEMES:
        conceptual_clips(theme);repair_clip(theme);architecture(theme);comparison(theme)
    print('Saved transparent themed clips and native diagrams. Numerical source files unchanged.')
