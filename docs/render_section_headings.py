"""Transparent section titles with an outlined font and a fine colored rule.

Requires fonttools. Pass a locally licensed display font with --font.
No font binary is distributed; only the resulting title artwork is saved.
"""
from pathlib import Path
import argparse,re,html,json
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--font',required=True)
args=parser.parse_args()
font=TTFont(args.font)
if 'fvar' in font:
    axes={a.axisTag:a for a in font['fvar'].axes};loc={}
    for tag,value in [('wght',550),('wdth',95)]:
        if tag in axes:loc[tag]=max(axes[tag].minValue,min(value,axes[tag].maxValue))
    font=instantiateVariableFont(font,loc,inplace=False)
glyphs=font.getGlyphSet();cmap=font.getBestCmap();units=font['head'].unitsPerEm
out=ROOT/'docs/assets/headings';out.mkdir(parents=True,exist_ok=True)
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
headings=re.findall(r'^(#{2,3}) (.+)$',text,re.M)
registry=out/'HEADINGS.json'
if not headings:
    headings=json.loads(registry.read_text(encoding='utf-8'))
else:
    registry.write_text(json.dumps(headings,indent=2),encoding='utf-8')
assert len(headings)==14,len(headings)

def svg_path(label,size):
    scale=size/units;x=0;paths=[]
    for char in label:
        name=cmap.get(ord(char),'.notdef');pen=SVGPathPen(glyphs)
        transform=TransformPen(pen,(scale,0,0,-scale,x,34))
        glyphs[name].draw(transform);paths.append(pen.getCommands())
        x+=font['hmtx'][name][0]*scale
    assert x<875,(label,x)
    return ''.join(paths)

for marks,title in headings:
    slug=re.sub(r'\s+','-',re.sub(r'[^a-z0-9\s-]','',title.lower()))
    small=len(marks)==3;path=svg_path(title,25 if small else 31)
    for theme,color,teal,amber in [('light','#14364b','#157b88','#b77c24'),('dark','#eef5fa','#6dd9dc','#f3c16c')]:
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="62" viewBox="0 0 900 62" role="img" aria-label="{html.escape(title,quote=True)}">
<title>{html.escape(title)}</title>
<defs><linearGradient id="rule" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="900" y2="0"><stop offset="0" stop-color="{teal}"/><stop offset="0.55" stop-color="{teal}" stop-opacity="0.7"/><stop offset="0.78" stop-color="{amber}" stop-opacity="0.55"/><stop offset="1" stop-color="{amber}" stop-opacity="0"/></linearGradient></defs>
<path fill="{color}" d="{path}"/>
<path d="M0 50.5H900" stroke="url(#rule)" stroke-width="1.2"/>
</svg>'''
        (out/f'{slug}-{theme}-v2.svg').write_text(svg,encoding='utf-8')
    level=4 if small else 3
    block=f'''<a id="{slug}"></a>
<h{level}><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headings/{slug}-dark-v2.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/headings/{slug}-light-v2.svg">
  <img src="docs/assets/headings/{slug}-light-v2.svg" alt="{html.escape(title,quote=True)}" width="900">
</picture></h{level}>'''
    text=text.replace(marks+' '+title,block,1)
# Keep the first technical definition readable without a long italic span.
text=text.replace('*Missingness-Informed Recognition of Activities from Wi-Fi CSI*','Missingness-Informed Recognition of Activities from Wi-Fi CSI')
readme.write_text(text,encoding='utf-8')
print(f'Rendered {len(headings)} section titles in two themes; no H2 grey border remains.')
