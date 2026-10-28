# Attribution and scientific precedents

MIRA-CSI spectral variant E (experimental), developed by Abdessamad Bourkibate,
2026. This package extends the MIRA-CSI research implementation and retains
its GPLv3 licence. The execution-specific contribution is the explicitly
defined observed-endpoint lagged-increment normalization, the model/control
implementation, and the reproducible exploratory reliability comparison.
No first-in-field or legal patent claim is made.

Established ingredients and primary sources:

- Qian, Wu, Yang, Liu and Zhou, **PADS**, ICPADS 2014. Covariance/eigenvalue
  features of normalized CSI amplitude/phase for binary motion detection.
  [DOI](https://doi.org/10.1109/PADSW.2014.7097784),
  [author paper](https://cswu.me/papers/ICPADS14_PADS_paper.pdf), Eq. 10 and
  subsequent text. It is not a task/split-matched seven-class comparator.
- Wiskott and Sejnowski, **Slow Feature Analysis**, Neural Computation 14(4),
  715–770, 2002. Covariance-normalized temporal-change features.
  [DOI](https://doi.org/10.1162/089976602317318938).
- Belouchrani, Abed-Meraim, Cardoso and Moulines, **SOBI**, IEEE TSP 45(2),
  434–444, 1997. Second-order temporal covariance and joint diagonalization.
  [DOI](https://doi.org/10.1109/78.554307).
- Che, Purushotham, Cho, Sontag and Liu, **GRU-D**, Scientific Reports 8:6085,
  2018. Missingness masks and time-interval information are known inputs.
  [DOI](https://doi.org/10.1038/s41598-018-24271-9). The present distance is
  symmetric/offline and is not GRU-D's decay mechanism.
- Liu et al., **Partial Convolutions**, ECCV 2018, 85–100. Observed-support
  normalized filtering has prior art; the present CNN does not implement that
  paper's mask-update operator. [Official proceedings](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.php).
- Guo, Pleiss, Sun and Weinberger, **Temperature Scaling**, ICML/PMLR 70,
  1321–1330, 2017. [Official proceedings](https://proceedings.mlr.press/v70/guo17a.html).

No third-party model source from those papers is bundled. Their attribution
describes prior methods rather than an invented implementation licence.
PyTorch, NumPy, SciPy and threadpoolctl are dependencies installed separately
and retain their own licences. The upstream dataset/parsing attribution and
MIT notice are retained separately in `DATA-ATTRIBUTION.md` and `licenses/`.
