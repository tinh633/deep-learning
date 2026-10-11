# FashionMNIST explorer

- Scope: Python preparation and one HTML page using Live Server. No training yet.
- Source: author's original IDX files verified against published MD5.
- Layout: .py in src/, one HTML in visualization/, shallow data/FashionMNIST_raw/, statistics/screenshots in outputs/.
- HTML loads EDA JSON and extracted IDX asynchronously through relative HTTP paths; no embedded base64 or separate template.
- Browser validates input and displays progress/errors. Serve workspace/project root.
- Virtual gallery bounds DOM while allowing access to every filtered image.
- Full-dataset EDA includes exact duplicate checks; no claim of near-duplicate detection or annotation correctness.
- Actual Edge HTTP smoke test passed gallery/EDA, pixels, filters and final-image details with no JS exceptions.
- Optional .cjs check files and temporary tools removed; validation evidence recorded here.
