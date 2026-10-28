"""Vector architecture figure of the implemented network, not sensing evidence."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

root=Path(__file__).resolve().parent
out=root/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(11.5,3.8));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.08',
                              facecolor=color,edgecolor='#45616E',linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9.4,linespacing=1.5)
def arrow(a,b,color='#45616E',style='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.3,color=color,linestyle=style))
box(.12,1.35,2.1,1.2,'18 amplitude bands\nTrain-only normalization\n500 packet positions','#E4EFFB')
box(2.8,1.35,1.7,1.2,'Concatenate\n18 + mask + gap\n20 input channels','#E8F3ED')
box(5.04,1.35,2.1,1.2,'Temporal CNN\n32 → 64 → 128\nk = 7, 5, 3; stride 2','#E0F0F4')
box(7.7,1.35,1.75,1.2,'Coverage-weighted\ntemporal pooling\n128 features','#FFF0D4')
box(10.0,1.35,1.75,1.2,'Dropout + linear\n7 activity logits\nScalar calibration','#EEE8F6')
for a,b in [((2.3,1.95),(2.68,1.95)),((4.58,1.95),(4.92,1.95)),((7.22,1.95),(7.58,1.95)),((9.53,1.95),(9.88,1.95))]:arrow(a,b)
box(2.8,2.98,2.1,.62,'Observed mask m; gap d','#FFF0D4')
arrow((3.65,2.9),(3.65,2.65),'#A06B16')
arrow((4.98,3.29),(8.57,3.29),'#A06B16')
arrow((8.57,3.29),(8.57,2.65),'#A06B16')
ax.text(6.7,3.52,'Mask → local coverage weights',ha='center',fontsize=9,color='#81530F')
ax.text(6,.79,'GroupNorm and GELU follow each convolution. Coverage is a deterministic proxy, not calibrated confidence.',ha='center',fontsize=9)
ax.text(6,.28,'CNN comparators: clean training or source-only loss augmentation. MIRA-CSI: 40,871 parameters.',ha='center',fontsize=9)
fig.tight_layout(pad=.5)
fig.savefig(out/'mira_architecture.pdf',bbox_inches='tight')
fig.savefig(out/'mira_architecture.png',dpi=200,bbox_inches='tight')
print('Saved implemented-model diagram.')
