# Prior art and attribution

MIRA-CSI is an authored implementation and reproducible evaluation of a compact
missingness-informed CSI classifier. It does not introduce masks, gap encodings,
CNNs, calibration or missing-CSI recovery as general concepts. The controlled
follow-up does not establish that all three additions are individually helpful.
This record is focused, not an exhaustive novelty assessment.

## Missing observations and temporal information

**Che, Z., Purushotham, S., Cho, K., Sontag, D., and Liu, Y. (2018).**
*Recurrent Neural Networks for Multivariate Time Series with Missing Values.*
Scientific Reports 8, 6085.
[Published article](https://doi.org/10.1038/s41598-018-24271-9).
GRU-D uses observation masks, elapsed intervals and decay mechanisms. MIRA's
nearest-support distance and coverage weighting differ in implementation; the
general use of missingness and temporal information is established prior work.

**Zhao, Z., Chen, T., Meng, F., Li, H., Li, X., and Zhu, G. (2024).**
*Finding the Missing Data: A BERT-Inspired Approach Against Package Loss in Wireless Sensing.*
IEEE INFOCOM Workshops, pp. 1–6.
[Published DOI](https://doi.org/10.1109/INFOCOMWKSHPS61880.2024.10620769) ·
[Author manuscript](https://arxiv.org/abs/2403.12400) ·
[Official CSI-BERT code](https://github.com/RS2002/CSI-BERT).
CSI-BERT learns recovery of missing CSI through self-supervised bidirectional
representations. It establishes prior work on learned missing-CSI recovery.

**Zhao, Z., Meng, F., Lyu, Z., Li, H., Li, X., and Zhu, G. (2026).**
*CSI-BERT2: A BERT-Inspired Framework for Efficient CSI Prediction and Classification in Wireless Communication and Sensing.*
IEEE Transactions on Mobile Computing 25(5), 7241–7257.
[Published DOI](https://doi.org/10.1109/TMC.2025.3640420) ·
[Author manuscript, revised 2025](https://arxiv.org/abs/2412.06861) ·
[Official code](https://github.com/RS2002/CSI-BERT2).
It combines pretraining and fine-tuning with adaptive subcarrier re-weighting,
temporal embeddings and masked prediction. The DOI contains 2025; the cited
volume/issue is 2026. These refer to the same published work.

**Zakeri, A., Nguyen, N. T., and Juntti, M. (2026).**
*Resource-Efficient Wi-Fi CSI-Based Sensing via Exploiting the Age of Samples.*
arXiv:2606.31690v2, revised 5 October 2026.
[Preprint](https://arxiv.org/abs/2606.31690v2) ·
[arXiv DOI](https://doi.org/10.48550/arXiv.2606.31690).
The authors encode retained-sample age relative to the sensing-session decision
and fuse it with CSI embeddings. This is a preprint citation; the arXiv DOI does
not by itself establish peer-reviewed publication. Sample age is not identical
to MIRA's offline distance to the nearest retained position, which can look ahead.

These CSI methods use different tasks, recordings and training access. Their
published scores are literature context, not matched numerical baselines in this
repository. No superiority over CSI-BERT, CSI-BERT2 or the age-aware model has
been tested here.

## Reference transform and dataset

**Dempster, A., Schmidt, D. F., and Webb, G. I. (2021).**
*MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification.*
KDD, 248–257.
[Published article](https://doi.org/10.1145/3447548.3467231) ·
[Official code](https://github.com/angus924/minirocket), pinned at
`0b1c245d9c9dbc50886f28bc7b32d5d45b5663d6`.
The evaluated basic multivariate transform produces 9,996 features. The 57.44%
clean source-selected result improves this known comparator's representation;
it is not a new MIRA architecture or a missing-input result.

**Alsaify, B. A., Almazari, M. M., Alazrai, R., and Daoud, M. I. (2020).**
*A dataset for Wi-Fi-based human activity recognition in line-of-sight and non-line-of-sight indoor environments.*
Data in Brief 33, 106534.
[Data article](https://doi.org/10.1016/j.dib.2020.106534) ·
[Dataset version 2](https://doi.org/10.17632/v38wjmz6f6.2).
The authors supply the underlying recordings. This repository's selected subset,
splits, synthetic loss operators and analyses do not transfer authorship of the
dataset. See [data attribution](data/README.md).

## Scope of the present evidence

The evaluated contribution is executable software plus a documented reliability
study and component controls. Its three-contrast B50-linear macro-F1 family is
exploratory and conditionally bootstrapped. Removing the distance input is slightly
better in this experiment; no adjusted standalone benefit for explicit mask input
or coverage weighting is established. Independent data and matched contemporary
recovery baselines are required for broader performance or novelty claims.
