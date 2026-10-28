# Attribution

Original MIRA-CSI implementation and research materials:
Copyright (c) 2026 Abdessamad Bourkibate, GPL-3.0-or-later.

The vendored MiniRocket reference remains credited to Angus Dempster,
Daniel F. Schmidt and Geoffrey I. Webb, under its retained GPL license.
The public dataset belongs to Baha' A. Alsaify, Mahmoud M. Almazari,
Rami Alazrai and Mohammad I. Daoud, under its separate CC BY 4.0 terms.
Author scaling code attribution/license is in data/SOURCE-CODE-LICENSE.txt.
Neither software authorship nor a model name transfers ownership of prior work.

## Spectral extension and methodological attribution

MIRA-CSI spectral variant E (experimental) is developed by Abdessamad Bourkibate.
The implemented observed-endpoint lagged-increment construction and its
exploratory comparison do not establish invention of covariance eigenspectra,
MLPs, CNNs or missingness inputs. PADS is earlier CSI covariance/eigen-feature
work; SFA and SOBI are precedents for covariance-normalized temporal statistics.
Their primary references and the exact attribution scope are retained in
[`spectral/NOTICE.md`](spectral/NOTICE.md). The included dataset derivatives
remain separately attributed under CC BY 4.0. Neither upstream ownership nor
endorsement is implied by this implementation.

## Author manuscript formatting class

`paper/source/mira-author.cls` is a **modified, renamed derivative of the
CAS 2.4 double-column class**, prepared for this author manuscript on
28 October 2026. Font selection was moved to `main.tex`; the current XeTeX
path uses `unicode-math` with TeX Gyre Pagella Math. The class records its
modification and new name; it is not represented as an unchanged CAS file.

The original `paper/source/cas-dc.cls` remains distributed unchanged, together
with the retained CAS support files, upstream README/manifest and original
licence notices. CAS files and this derivative retain the **LaTeX Project
Public License, version 1.2 or later**, as stated in their headers; those
terms remain separate from the repository's software GPL licence. The class
name and formatting do not confer journal acceptance or peer-review status.
