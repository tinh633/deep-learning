# Practice 3 — Folder tree and ownership

```text
practice_3_huggingface/
├── README.md                       # project overview
├── FOLDER_TREE.md                  # this file
├── exercise_1/
│   └── WORKFLOW.md                 # Exercise 1: independent inference workflow
├── exercise_2/
│   └── WORKFLOW.md                 # Exercise 2: preparation, training, evaluation
├── src/
│   ├── config.py                   # shared model/dataset/experiment settings
│   ├── utils.py, metrics.py        # shared utilities
│   ├── 0_eda.py                    # Exercise 2: data visualization
│   ├── 1_sentiment.py              # Exercise 1
│   ├── 2_preparedata.py            # Exercise 2
│   ├── 3_train.py                  # Exercise 2
│   ├── 4_evaluate.py               # Exercise 2
│   ├── 5_report.py                 # Exercise 2
│   └── 6_infer_finetuned.py        # Exercise 2: optional demo
├── docs/                           # design and theory for both exercises
├── tests/                          # unit tests; no generated pytest cache
├── outputs/                         # empty (contains only .gitkeep)
├── visualization/                   # empty (contains only .gitkeep)
├── Practice_3_Report.ipynb         # report notebook skeleton
├── Practice_3_Requirements.md
├── Experiment_Plan.md
├── Training.md
├── progress.md
└── requirements.txt
```

Source files remain under `src/` on purpose: they import `config`, `utils`, and `metrics` by sibling-module names. Moving them into exercise-specific folders without refactoring imports would break direct `python src/NAME.py` commands. Separating the Markdown workflows therefore provides distinct exercises without breaking working code.

**Starter cleanup:** removed `outputs/` contents, `visualization/` plots and generated run reports, `src/__pycache__/`, `tests/.pytest_cache/`. Retained source code, documentation and unexecuted notebook. Reset `EVAL_RUN='lan_1'` and `RUN_TEST=False` so no old run is referenced.
