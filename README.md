# Premier League Match Predictor

Learn ML by building a model that predicts Premier League results: **Home win (H)**, **Draw (D)**, or **Away win (A)**.

No notebooks — plain Python scripts you run from the terminal.

---

## Overview

| Piece | Details |
|--------|---------|
| **Goal** | Classify each match as H / D / A before kickoff |
| **Data** | `data/raw/results.csv` (~11k matches, seasons 1993–94 → 2021–22) |
| **Approach** | Clean data → baseline → features → train model → evaluate → predict |
| **Hard rule** | Only use info known **before** the match (no same-game goals/shots) |

Raw data is already **one row per fixture** (home team, away team, full-time score/result).

---

## Project layout

```
data/
  raw/results.csv       # original data — do not edit
  processed/            # cleaned / feature tables we write
src/                    # library code (importable functions)
  data.py               # load & clean
  features.py           # form, Elo, etc.
  model.py              # train / evaluate / save
  predict.py            # score a single fixture
scripts/                # things you run from the terminal
  explore.py
  prepare_data.py
  build_features.py
  train.py
  predict_match.py
requirements.txt
README.md
```

---

## How you’ll run things

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/explore.py
python scripts/prepare_data.py
python scripts/build_features.py
python scripts/train.py
python scripts/predict_match.py "Arsenal" "Chelsea"
```

Predictions use each team’s form/Elo **as of the last date in the dataset**. Refresh `data/raw/results.csv` and re-run prepare → features → train to update.

---

## What “good” looks like

- Beating the always-home baseline matters more than a fancy algorithm
- ~45–55% accuracy is a realistic range for 3-way football results; 80%+ usually means leakage or a bug
- Draws are hard — expect the model to struggle there at first
