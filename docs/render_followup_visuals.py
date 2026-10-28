"""Render real follow-up result summaries for light/dark GitHub appearances.

Run from the repository root: python docs/render_followup_visuals.py
Inputs are public summary CSVs under evaluation/component-controls/summaries.
No inference, fitting, synthetic curves, screenshots or session paths are used.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evaluation/component-controls/summaries'
OUT = ROOT / 'docs/assets'
MODELS = ['mira_full', 'no_mask_input', 'no_distance_input', 'uniform_pool',
          'resnet1d_clean', 'resnet1d_augmented']
LABELS = ['Full MIRA', 'Zero mask input', 'Zero distance input', 'Uniform pooling',
          'Residual CNN, clean', 'Residual CNN, augmented']
MARKERS = ['o', 's', 'D', '^', 'v', 'P']
THEMES = {
    'light': dict(text='#18344c', grid='#597387',
                  colors=['#00796B', '#BD4E00', '#0072B2', '#9862A8', '#526975', '#A78500'],
                  range='#738895', original='#526975', selected='#00796B'),
    'dark': dict(text='#dbe5ee', grid='#91a3b5',
                 colors=['#66D3C5', '#F4AA74', '#73B6EC', '#CC9CE0', '#BECBD4', '#E8CE6B'],
                 range='#91A4B3', original='#BECBD4', selected='#66D3C5'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style(theme):
    colors = THEMES[theme]
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'text.color': colors['text'], 'axes.labelcolor': colors['text'],
        'axes.titlecolor': colors['text'], 'xtick.color': colors['text'],
        'ytick.color': colors['text'], 'axes.facecolor': 'none',
        'figure.facecolor': 'none', 'savefig.transparent': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'legend.frameon': False, 'axes.titleweight': 'bold',
    })
    return colors


def save(fig, name, theme):
    target = OUT / f'followup-{name}-{theme}.png'
    fig.savefig(target, dpi=200, transparent=True)
    plt.close(fig)
    with Image.open(target) as image:
        assert image.mode == 'RGBA'
        alpha = np.asarray(image)[:, :, 3]
        assert np.all(alpha[[0, -1], :] == 0) and np.all(alpha[:, [0, -1]] == 0)
        return dict(file=target.relative_to(ROOT).as_posix(), width=image.width,
                    height=image.height, sha256=sha(target), transparent=True)


def model_comparison(summary, theme):
    colors = style(theme)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.6))
    groups = [('complete', 'none'), ('random', 'linear'), ('contiguous', 'linear')]
    for axis, metric, title, factor in zip(
            axes, ['accuracy', 'macro_f1', 'nll'],
            ['Accuracy (%)', 'Macro F1', 'Calibrated NLL'], [100, 1, 1]):
        for j, model in enumerate(MODELS):
            rows = [summary[(summary.model == model) & (summary.pattern == pattern)
                            & (summary.imputer == imputer)].iloc[0]
                    for pattern, imputer in groups]
            axis.errorbar(np.arange(3)+(j-2.5)*.045,
                          [r[metric+'_mean']*factor for r in rows],
                          yerr=[r[metric+'_std']*factor for r in rows],
                          color=colors['colors'][j], marker=MARKERS[j],
                          linewidth=1.7, markersize=5, capsize=2, label=LABELS[j])
        axis.set_xticks([0, 1, 2], ['C', 'R50', 'B50'])
        axis.set_xlim(-.27, 2.27)
        axis.set_title(title, fontsize=12, pad=12)
        axis.grid(axis='y', color=colors['grid'], alpha=.16)
        axis.tick_params(axis='both', length=0, labelsize=10, pad=7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(.5, .065), columnspacing=2)
    fig.text(.5, .025, 'C: complete  ·  R50/B50: random/contiguous 50% loss, linear repair  ·  mean ± sample SD',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.055, right=.985, top=.89, bottom=.30, wspace=.34)
    return save(fig, 'models', theme)


def component_intervals(contrasts, theme):
    colors = style(theme)
    names = ['no_mask_input', 'no_distance_input', 'uniform_pool']
    labels = ['Full − zero mask input', 'Full − zero distance input', 'Full − uniform pooling']
    frame = contrasts.set_index('comparator').loc[names]
    fig, axis = plt.subplots(figsize=(10.8, 4.2))
    for i, (_, row) in enumerate(frame.iterrows()):
        axis.plot([row.bonferroni98_333_low, row.bonferroni98_333_high], [i, i],
                  color=colors['range'], lw=6, solid_capstyle='butt')
        axis.plot([row.conditional95_low, row.conditional95_high], [i, i],
                  color=colors['selected'], lw=1.5)
        axis.plot([row.conditional95_low, row.conditional95_high], [i, i],
                  color=colors['selected'], marker='|', ms=14, mew=1.5, linestyle='none')
        axis.plot(row.difference, i, 'o', color=colors['text'], ms=7)
    axis.axvline(0, color=colors['grid'], alpha=.7, ls=(0, (3, 3)), lw=1)
    axis.set_yticks(range(3), labels)
    axis.set_ylim(2.5, -.5)
    axis.tick_params(axis='both', length=0, pad=10)
    axis.grid(axis='x', color=colors['grid'], alpha=.14)
    axis.set_xlabel('Macro-F1 difference  ·  positive favours full MIRA', labelpad=12)
    axis.set_title('Component controls · contiguous 50% loss, linear repair', fontsize=13, pad=15)
    fig.legend([Line2D([0], [0], color=colors['selected'], lw=1.5, marker='|', ms=10),
                Line2D([0], [0], color=colors['range'], lw=6)],
               ['Conditional 95%', 'Bonferroni 98.333% · three contrasts'],
               loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(.52, .025))
    fig.subplots_adjust(left=.285, right=.985, top=.84, bottom=.27)
    return save(fig, 'components', theme)


def minirocket_clean(per_fold, summary, theme):
    colors = style(theme)
    fig, axis = plt.subplots(figsize=(10.8, 4.2))
    for variant, label, color, marker, offset in [
            ('identity_control', 'Original representation', colors['original'], 'o', -.08),
            ('source_selected', 'Source-selected representation', colors['selected'], 'D', .08)]:
        values = []
        for fold in [1, 2, 3]:
            score = per_fold[(per_fold.variant == variant) & (per_fold.held_environment == fold)].accuracy
            assert len(score) == 5
            values.append((score.mean()*100, score.std(ddof=1)*100))
        pooled = summary[summary.variant == variant].iloc[0]
        values.append((pooled.accuracy_mean*100, pooled.accuracy_sd*100))
        axis.errorbar(np.arange(4)+offset, [x[0] for x in values],
                      yerr=[x[1] for x in values], color=color, marker=marker,
                      linestyle='none', markersize=7, capsize=4, linewidth=1.7, label=label)
    axis.axvline(2.5, color=colors['grid'], alpha=.25, ls=(0, (3, 3)), lw=1)
    axis.set_xticks(range(4), ['Excluded E1', 'Excluded E2', 'Excluded E3', 'Pooled held'])
    axis.set_ylabel('Clean accuracy (%)')
    axis.set_title('MiniRocket comparator · all three domains and pooled results', fontsize=13, pad=15)
    axis.tick_params(axis='both', length=0, pad=10)
    axis.grid(axis='y', color=colors['grid'], alpha=.16)
    axis.set_xlim(-.4, 3.4)
    axis.legend(loc='upper right', fontsize=10)
    fig.text(.5, .035, 'Fold-specific source-validation selection  ·  five seeds  ·  mean ± sample SD  ·  clean inputs only',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.23)
    return save(fig, 'minirocket', theme)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = ['held-summary.csv', 'primary-component-contrasts.csv',
              'minirocket-clean-per-fold-seed.csv', 'minirocket-clean-summary.csv']
    summary, contrasts, per_fold, clean = [pd.read_csv(DATA/name) for name in inputs]
    assert len(summary) == 42 and len(contrasts) == 3 and len(per_fold) == 30 and len(clean) == 2
    assert set(summary.model) == set(MODELS)
    files = []
    for theme in THEMES:
        files.extend([model_comparison(summary, theme), component_intervals(contrasts, theme),
                      minirocket_clean(per_fold, clean, theme)])
    provenance = dict(
        source='Saved real-evaluation summary CSVs; no illustrative or generated experimental values.',
        files=files,
        input_sha256={f'evaluation/component-controls/summaries/{name}':sha(DATA/name) for name in inputs},
        renderer='docs/render_followup_visuals.py', renderer_sha256=sha(__file__),
        source_models=6, source_component_fits=90, clean_reference_selected_fits=30,
        uncertainty='Sample SD across five training seeds after averaging masks; component intervals use the separate fixed-model participant bootstrap.',
        distinction='MiniRocket source-selected results are clean-only and are not attributed to MIRA.',
        backgrounds='RGBA transparent in both themes; no external frame or fake interface.')
    (OUT/'followup-provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(files, indent=2), flush=True)


if __name__ == '__main__':
    main()
