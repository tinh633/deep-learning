# AI collaboration record

- Commit approval: user explicitly requested committing prepared changes on Thái. Dataset/cache/secrets remain excluded; push not authorized. Final commit hash is reported in chat and can be verified with git log.

- Git preparation: connected workspace root to tinh633/deep-learning, created local branch Thái from origin/main, added root .gitignore excluding downloaded datasets/cache/secrets and integrated root README. No new commit or push; awaits user approval.

- Source-link update: saved the actual four download URLs and checksums in docs/data_source.md; linked README/preparation docs and updated reusable documentation rules. No dataset or code changes.

## 2026-10-11 — Data download and explorer

- User contribution: supplied Practice 1 assignment; chose Python scripts rather than notebook; requested full-label image browsing and detailed EDA in HTML.
- User modifications: none observed in newly generated project files during this task.
- Agent actions: inspected local rules and dependencies, verified source information, wrote Python preparation script, HTML template, integration checks and documentation.
- Tools: shell/Python for download and statistics; Node for JavaScript verification; web for official source/checksums; Sites skill for local HTML workflow. No publication or messaging tools used.
- Generated components: all files in this project except original dataset files downloaded from the author.
- Request purpose: inspect data before model development.
- Validation: four MD5 checksums passed; 33 gallery filters and final-image access passed; three EDA views passed; torchvision independently loaded both splits and matched counts and pixel means.
- Failure resolved: torchvision initially could not find extracted IDX files; added extraction from verified compressed originals, then validation passed.
- Results: 60,000 train / 10,000 test; balanced classes; no exact duplicate pixel arrays found.
- Limitations: no visual browser QA; no near-duplicate or manual label audit; no model training yet.
- Approval state: implementation requested by user; completed result awaits review. No commit or push.

## Layout revision — Part 2

- User requested continuation after review of updated file-placement rules.
- Moved HTML template and generated page into `visualization/`; flattened data to `data/FashionMNIST_raw/`.
- Updated preparation/verification paths and project documentation. No HTML behavior changes.
- Verified SHA-256 unchanged for all eight moved data files; preparation passed four MD5 checks without downloading again; Node integration checks passed.
- Previous torchvision validation applied to the old cache layout. The new shallow directory is not its default cache layout.
- HTML viewing issue remains pending for Part 3. No commit or push.

## HTML viewing revision — Part 3

- Diagnosed a concrete issue: the open `explorer.html` was a template with unresolved placeholders, which cannot parse as dataset JSON.
- Renamed source to `visualization/explorer_template.html`; actual data viewer is now `visualization/explorer.html`. Updated generation and verification paths.
- Added explicit Windows/VS Code opening instructions; source template is not a directly viewable dataset page.
- Actual headless Edge test passed gallery/EDA rendering, nonblank pixel canvas, 11 label options, test/class filtering and final-image dialog, with no JavaScript exceptions.
- Inspected screenshot `outputs/browser_preview.png`; gallery and controls are visible. Node 33-filter regression check also passed.
- User desktop viewing confirmation remains pending. No model training, commit or push.

## Python-only src revision

- User clarified reusable layout preference: src contains only Python files; visual checks live with visualization assets.
- Moved verify_explorer.cjs and browser_check.cjs from src/ to visualization/; updated commands, script paths, design, and reusable agent rules.
- Validation: 33-filter Node integration checks passed from the new location; browser-check JavaScript syntax passed. src contains only prepare_data.py.
- Browser smoke test was not rerun for this relocation; its project-root calculation is unchanged. Prior Edge results remain historical evidence.
- User confirmed assignment uses MLP but explicitly deferred model work. No training, commit or push.

## Single HTML viewer revision

- User requested one HTML and reported no dataset in VS Code Preview.
- Removed redundant explorer_template.html; preparation now refreshes embedded data in explorer.html while preserving UI code and validating block counts.
- Updated reusable rules and active documentation; added visible guidance when the loading message persists in IDE Preview.
- Validation: preparation checksum checks passed; 33-filter integration checks passed; actual Edge test passed rendered gallery/EDA, nonblank canvas and detail/filter interactions without JS exceptions.
- VS Code Preview remains unverified; its specific restriction has not been established. Instructions use verified Edge/Chrome opening rather than claiming Preview is fixed.
- No dataset deletion, training, commit or push.

## Live Server revision

- User requested Live Server viewing first and removal of unnecessary .cjs files.
- Replaced embedded base64 dataset with async fetch of EDA JSON and original extracted IDX files; HTML reduced from approximately 70 MiB to 18 KB.
- Added loading/error messages, HTTP requirement, input validation and disabled controls during initialization.
- Python now prepares only dataset and EDA, preserving the HTML UI.
- Preparation passed four MD5 checks. Actual headless Edge over local HTTP passed gallery/EDA, nonblank canvas, class/test filter and final-image dialog with no JS exceptions; screenshot refreshed.
- Removed both project .cjs scripts, their paired docs and temporary verification tools. No Node required for project use.
- Updated reusable layout rules, design and Live Server instructions. Extension itself not automated; user viewing confirmation pending.
- No data deletion, training, commit or push.
