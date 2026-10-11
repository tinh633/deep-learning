# prepare_data.py

- Purpose: download/verify FashionMNIST and prepare files for the HTTP viewer.
- Inputs: official compressed IDX files; optional --download-only.
- Source links and checksums: [data_source.md](../data_source.md).
- Outputs: originals and extracted IDX in data/FashionMNIST_raw/, outputs/eda.json.
- Dependencies: Python standard library and NumPy.
- Validation: MD5, headers, counts, dimensions and label range.
- EDA: pixel moments/histograms, brightness quantiles, class distribution/mean images, foreground ratio, blank/constant images, SHA-256 exact duplicates.
- Run: `python practice_1_fashionmnist/src/prepare_data.py`.
- Does not generate or rewrite HTML. The browser fetches the prepared files via Live Server.
- Preparation and actual Edge HTTP consumption passed; network required for initial download.
- Exact duplicate checks do not detect near duplicates or establish label correctness.
