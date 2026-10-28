# Data attribution and processing

Original acquisition: Baha' A. Alsaify, Mahmoud M. Almazari, Rami Alazrai and
Mohammad I. Aldaoud. **A dataset for Wi-Fi-based human activity recognition in
line-of-sight and non-line-of-sight indoor environments**, Mendeley Data v2,
16 February 2021, [DOI 10.17632/v38wjmz6f6.2](https://doi.org/10.17632/v38wjmz6f6.2).
The [primary dataset page](https://data.mendeley.com/datasets/v38wjmz6f6/2)
specifies **CC BY 4.0**; its licence text is in `licenses/CC-BY-4.0.txt`.
The data article is [DOI 10.1016/j.dib.2020.106534](https://doi.org/10.1016/j.dib.2020.106534).

Original parsing/scaling repository:
[lcsig dataset repository](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments),
pinned commit `5337147c5156948b13e6b05a50b79529614acfbe`.
Its source-software MIT notice is retained in `licenses/SOURCE-SOFTWARE-MIT.txt`.
Original acquisition and author software are not claimed as MIRA inventions or
as endorsement of this derivative implementation.

Changes: immutable selection and participant roles; original author/Halperin
scaling; mean squared scaled amplitude over five consecutive exported
subcarriers, followed by `10*log10(max(power,1e-24))`; three receiving antennas
and six groups per antenna; linear interpolation to 500 valid-packet-index
positions. These values are **dB of dimensionless scaled power**, not dBm,
physical seconds, body temperature or a reconstructed human image. Source
mean/std normalization is not fitted into this fixed cache. Static per-record
preprocessing and cache hash reads do not constitute target model fitting.

`data/manifest.json` records the 1,260 selected recordings and roles.
`data/power_representation_manifest.json` records all raw-record hashes, the
fixed rule and cache lineage. The optional training-input asset contains this
derived cache, not the complete original dataset. Its SHA256 is
`e418b7e7061697f4100b7a071b2478226761ff614eb6a97ae482997593353523`.

`demo/provenance.json` identifies the included example derived under CC BY 4.0. It is
the first held E1 recording in manifest order, selected without consulting
predictions; both errors are retained. Cite the original dataset and identify
these transformations when redistributing the example or cache. Data remain
separately attributed under CC BY 4.0; the software licence does not replace
the dataset attribution or grant ownership of the original recordings.
