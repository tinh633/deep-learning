# explorer.html

- Only visualization HTML: visualization/explorer.html, approximately 18 KB.
- Open with Live Server from workspace/project root. JavaScript is required; file:// and editor source Preview are not supported.
- Inputs: ../outputs/eda.json and four extracted IDX files in ../data/FashionMNIST_raw/ through relative HTTP paths.
- Async initialization displays loading/errors; filters remain disabled until ready.
- Validates IDX shape, count, size, labels and report count.
- Features: all-data/class/split filters, exact counts, virtual gallery, position jump, details, split-specific EDA.
- Approximately 55 MB data loads once per page load. No external JS libraries or .cjs files needed.
- Actual Edge HTTP checks passed nonblank pixels, gallery/EDA, class/test filters and final-image dialog, without JS exceptions.
- Live Server extension itself was not controlled by automation.
