# Public derived CSI input

The 18 × 500 tensors derive from the public dataset by Baha' A. Alsaify,
Mahmoud M. Almazari, Rami Alazrai and Mohammad I. Daoud:
https://doi.org/10.17632/v38wjmz6f6.2 (version 2, CC BY 4.0).
Data article: https://doi.org/10.1016/j.dib.2020.106534.

The dataset authors retain ownership and credit. The software license does not
replace the dataset's separate CC BY 4.0 license:
https://creativecommons.org/licenses/by/4.0/.

Processing applied for this study: author/Halperin sqrt(SNR) amplitude scaling,
20 log10 amplitude with a 1e-12 floor, five-subcarrier averaging into six bands
per receive antenna, and interpolation to 500 normalized packet positions.
The immutable file selection/order and raw-file SHA256 hashes are recorded in
`manifest.json`. Derived tensor SHA256 is recorded in `../input_receipt.json`.

The optional `mira-csi-public-inputs.zip` release asset contains only this input
cache and its provenance. Extract it into the project directory to obtain
`data/band_streams.npy`; it does not overwrite the current software source.
Alternatively, `python prepare_data.py` reconstructs the cache from the pinned
public raw recordings and checks the exact expected tensor hash.
