# NFL Big Data Bowl 2027: Frame First

**What the Combine can and cannot tell you about a receiver's separation**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-007acc.svg)](https://docs.pytest.org/)

An entry for the **[NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)** on Kaggle, which asks how movement at the Scouting Combine translates to NFL play.

---

## 🏆 Deliverables

- 📄 **Submission write-up:** [`docs/SUBMISSION_WRITEUP.md`](docs/SUBMISSION_WRITEUP.md). Every statistic in it is printed by the notebook, and `tests/test_writeup_provenance.py` fails if one is not.
- 📓 **Kaggle notebook:** [`notebooks/frame_first.ipynb`](notebooks/frame_first.ipynb). Self-contained and the single source of truth. The six write-up figures in [`docs/`](docs/) (`fig1_*` to `fig6_*`) are its cell outputs.
- 🔎 **Field survey:** [`docs/research/2026-10-13-writeup-survey.md`](docs/research/2026-10-13-writeup-survey.md). What 178 public notebooks do, and the probes that led to this direction.
- 🧪 **Legacy pipeline:** [`src/bdb_2027/`](src/bdb_2027/) belongs to an earlier multi-position analysis. It uses different cohorts and feature definitions and is not what the write-up reports.

---

## 🔬 What the notebook finds

1. **Frame predicts separation.** For wide receivers, weight and height predict separation at the throw in a draft class the model never saw. The official 40 time and draft slot do not.
2. **The sensors add little to that.** Tracked top speed adds almost nothing, and position-drill tracking makes the forecast worse, because single drill reps repeat poorly.
3. **Size costs the most on screens and crossing routes**, and against zone coverage. It costs the least on slants and outs.
4. **Play speed is the pathway.** On the same route with the same time to throw, lighter receivers reach a higher speed before the ball comes out. Tracked Combine speed predicts that play speed where the official 40 time does not.
5. **Frame beats early tape.** It forecasts a receiver's later separation better than his own first 200 routes.
6. **The trenches echo it.** Weight explains what the hoop drill appears to say about get-off, and nothing at the Combine relates to pressure probability allowed.

Numbers, intervals and limits are in the write-up.

---

## 📁 Repository Structure

```
.
├── .agents/skills/                    # Installed agent skills (literature, kaggle, etc.)
├── data/
│   ├── raw/                           # Raw competition CSV files (2.2 GB)
│   ├── discussions.db                 # SQLite cache of Kaggle forum discussions
│   └── kernels.db                     # SQLite cache of top competitor kernels
├── docs/
│   ├── SUBMISSION_WRITEUP.md          # Competition submission write-up
│   ├── fig1_* ... fig6_*              # Write-up figures, exported from the notebook
│   └── research/                      # Field survey and re-runnable direction probes
├── notebooks/
│   ├── frame_first.ipynb              # Kaggle submission notebook, single source of truth
│   └── kernel-metadata.json           # Kaggle CLI kernel deployment manifest
├── references/
│   ├── literature_and_resources.md    # Comprehensive literature registry
│   └── bibliography.json              # Machine-readable citations database
├── src/bdb_2027/
│   ├── config.py                      # Pydantic configuration & paths
│   ├── datasets/
│   │   ├── kinematics.py              # 10 Hz tangential & centripetal decomposition
│   │   ├── combine.py                 # Combine tracking & drill feature extractor
│   │   └── game.py                    # player_play.csv outcome aggregation
│   ├── modeling/
│   │   └── pipeline.py                # GroupKFold Ridge & LightGBM translation models
│   └── scripts/
│       ├── download_data.py           # Automated Kaggle data fetcher
│       └── run_translation_analysis.py# Full pipeline orchestration
└── tests/
    ├── test_notebook_frame_first.py   # Runs the notebook's helper cell on synthetic inputs
    ├── test_writeup_provenance.py     # Every statistic in the write-up must be printed by the notebook
    └── test_bdb_pipeline.py           # Legacy pipeline test suite
```

---

## 🚀 Quick Start

### 1. Environment Setup

Using [`uv`](https://astral.sh/uv):

```bash
uv sync
```

### 2. Download Competition Data

```bash
uv run download-data
```

### 3. Run Pipeline Unit Tests

```bash
uv run pytest tests/test_bdb_pipeline.py
```

### 4. Run End-to-End Translation Analysis

```bash
uv run python src/bdb_2027/scripts/run_translation_analysis.py
```

---

## 🤖 Kaggle Kernel Deployment

The submission notebook [`notebooks/frame_first.ipynb`](notebooks/frame_first.ipynb) is self-contained and pre-configured with `notebooks/kernel-metadata.json`.

To push or submit to Kaggle:

```bash
# Push kernel to Kaggle
kaggle kernels push -p notebooks/
```

Or execute via the bundled NVIDIA Kaggle Skill:

```bash
PYTHONPATH=.agents/skills/nvidia-kaggle-skill/scripts uv run python .agents/skills/nvidia-kaggle-skill/scripts/submit_kernel.py notebooks/
```

---

## 📄 License & Attribution

This project is licensed under the MIT License. External literature and scientific frameworks referenced are credited in [`references/literature_and_resources.md`](references/literature_and_resources.md).
