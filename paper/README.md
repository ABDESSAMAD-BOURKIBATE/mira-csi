# Research manuscript

**Reliability Assessment of Wi-Fi CSI-Based Activity Recognition under Environmental Shift and Structured Measurement Loss**<br>
Eng. **Abdessamad Bourkibate** · 28 October 2026

[Read the complete PDF](manuscript.pdf) · [Four-page result supplement](supplementary-results.pdf) · [LaTeX source](source/main.tex) · [File hashes](MANIFEST.json) · [Spectral software](../spectral/README.md)

This is the **unreviewed author manuscript**: **32 pages, 19 tables, 18 figures
and 25 references**, with a separate **four-page A–E result supplement**. It has
not been accepted by a journal and has **no assigned DOI**. It retains the
author-approved AI-assistance declaration and the negative results from the
earlier studies. The [v0.2.0 archived PDF](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.2.0/MIRA-CSI-reliability-manuscript-20261028.pdf)
is an earlier author version; its published checksums continue to identify
that archived file, not this expanded version.

The current version adds the 75-fit spectral A–E extension and an additional
15-configuration MiniRocket power18 reference (three source fits reused,
12 additional source fits). It reports the feature-only E result alongside
the weaker hybrids and the small original A−B contrast. The E/reference
comparison is post-hoc and uses different augmentation/mean-fill policies;
it does not isolate architecture or support matching or establish SOTA,
patent novelty or independent confirmation. All uncertainty ranges retain
their fixed-data/room/model scope.

<p align="center"><a href="manuscript.pdf"><img src="first-page.png" alt="Actual first page of the MIRA-CSI reliability manuscript" width="480"></a></p>

## What is included

| File | Purpose |
|---|---|
| [manuscript.pdf](manuscript.pdf) | Complete compiled author version |
| [supplementary-results.pdf](supplementary-results.pdf) | Four pages retaining all 90 A–E fold/domain/condition rows, including unfavourable familiar results |
| [source/](source/) | Main TeX file, included result fragments, referenced figures, modified author class and unchanged CAS support files |
| [spectral-results-page.png](spectral-results-page.png) | Current rendered manuscript page with spectral A–E results |
| [matched-reference-page.png](matched-reference-page.png) | Current rendered manuscript page with the power18 reference comparison |
| [results-page.png](results-page.png) · [component-page.png](component-page.png) | Retained earlier author-version result previews |
| [MANIFEST.json](MANIFEST.json) | SHA256 values tying this source package to the PDF |

The original 60-fit study, 90-fit component study, clean-input MiniRocket
representation sensitivity, 75-fit A–E extension and later power18 reference
retain distinct protocols and numerical artifacts. Reused source fits are not
new independent data. Read the captions before comparing paired-mask summaries
with full-mask-grid summaries, or different representations/training policies.

The v0.3.0 release assets use these exact names:

- [`MIRA-CSI-completed-A-E-spectral-evidence-20261028-validated.zip`](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-completed-A-E-spectral-evidence-20261028-validated.zip)
- [`MIRA-CSI-matched-power-MiniRocket-evidence-20261028.zip`](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-matched-power-MiniRocket-evidence-20261028.zip)
- [`MIRA-CSI-spectral-E-package.zip`](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-spectral-E-package.zip), [`MIRA-CSI-spectral-E-weights.zip`](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-spectral-E-weights.zip) and [`MIRA-CSI-spectral-training-inputs.zip`](https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/releases/download/v0.3.0/MIRA-CSI-spectral-training-inputs.zip)

The manuscript discloses room/cohort confounding, synthetic masking after packet
normalization, offline future-aware repair, conditional uncertainty and known
historical target results. Body meshes and wavefront illustrations explain
concepts; they are not measured poses, thermal images or solved RF fields.

## Compile the source

Open `source/` as a LaTeX project and select `main.tex`. Use **XeLaTeX** on
Overleaf or locally, or **Tectonic** (XeTeX). The current source uses
`unicode-math` and TeX Gyre Pagella Math on that engine path, including the
matrix and logical symbols. From `paper/source/`, for example:

```sh
latexmk -xelatex -interaction=nonstopmode main.tex
```

Or with Tectonic:

```sh
tectonic -X compile --untrusted main.tex
```

The local Tectonic build was checked. Package/font versions can change line
breaks and page counts; the distributed PDF and [MANIFEST.json](MANIFEST.json)
identify the checked version. These commands compile the main manuscript;
the supplementary-results PDF is supplied separately.

The main document selects **`mira-author.cls`**, a renamed, modified derivative
of the CAS 2.4 double-column class. Its modification notice records that font
selection moved to `main.tex`. The original `cas-dc.cls` remains unchanged and
distributed alongside the CAS style/bibliography files, original notices and
upstream README/manifest. The class and derivative retain the **LaTeX Project
Public License, version 1.2 or later**, as stated in their headers. They are
separate from the software GPL licence. Dataset and scaling-code rights remain
separate as recorded in [NOTICE.md](../NOTICE.md) and [data attribution](../data/README.md).

## Cite this author version

```bibtex
@unpublished{bourkibate2026reliability,
  author = {Bourkibate, Abdessamad},
  title = {Reliability Assessment of Wi-Fi CSI-Based Activity Recognition under Environmental Shift and Structured Measurement Loss},
  year = {2026},
  note = {Unreviewed author manuscript, 28 October 2026},
  url = {https://github.com/ABDESSAMAD-BOURKIBATE/mira-csi/tree/main/paper}
}
```

Cite the original dataset and MiniRocket separately when using them. Versioned
source and SHA256 records identify the exact author version distributed here.
