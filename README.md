# Premier League Match Predictor

Learn ML by building a model that predicts Premier League results: **Home win (H)**, **Draw (D)**, or **Away win (A)**.

No notebooks — plain Python scripts you run from the terminal.

---

## Overview

| Piece | Details |
|--------|---------|
| **Goal** | Classify each match as H / D / A before kickoff |
| **Data** | `data/raw/matches.csv` (~1,389 team-rows, seasons 2020–22) |
| **Approach** | Clean data → baseline → features → train model → evaluate → predict |
| **Hard rule** | Only use info known **before** the match (no same-game goals/xG/shots) |

Raw rows are one team’s view of a game. Each fixture appears twice; we merge into **one row per match** before modeling.

---

## Project layout

```
data/
  raw/matches.csv       # original data — do not edit
  processed/            # cleaned / feature tables we write
src/                    # library code (importable functions)
  data.py               # load & clean
  features.py           # form, Elo, etc.
  model.py              # train / evaluate / save
  predict.py            # score a single fixture
scripts/                # things you run from the terminal
  explore.py            # print data summaries
  prepare_data.py       # raw → processed matches
  train.py              # train + evaluate
  predict_match.py      # CLI: predict Home vs Away
requirements.txt
README.md
```

- **`src/`** = reusable logic  
- **`scripts/`** = entry points (`python scripts/train.py`)

---

## Build plan (do in order)

### Phase 1 — Understand the data
- Run exploration: shape, date range, teams, result balance
- Confirm home/away pairing works
- **Deliverable:** `scripts/explore.py`

### Phase 2 — Clean into match-level table
- Merge home + away rows → one fixture per row
- Columns like: `date`, `home_team`, `away_team`, `home_goals`, `away_goals`, `result`
- Save to `data/processed/matches.csv`
- **Deliverable:** `src/data.py` + `scripts/prepare_data.py`

### Phase 3 — Baseline (no ML yet)
- Always predict “Home win”
- Score with a **time-based** split (train on earlier games, test on later)
- This is the score every real model must beat
- **Deliverable:** printed baseline accuracy in `scripts/train.py` (or a small baseline step)

### Phase 4 — Features
- Rolling form (e.g. points / GD last 5 games)
- Optional: Elo ratings, season standing so far
- Still only pre-match info
- Save feature table if useful: `data/processed/features.csv`
- **Deliverable:** `src/features.py`

### Phase 5 — Train a model
- Start with **logistic regression** (simple, readable)
- Later try Random Forest if you want
- Time-based train/test (never shuffle randomly for football)
- Metrics: accuracy vs baseline, confusion matrix, maybe log loss
- **Deliverable:** `src/model.py` + `scripts/train.py`

### Phase 6 — Predict
- CLI: `python scripts/predict_match.py "Arsenal" "Chelsea"`
- Print H / D / A probabilities
- **Deliverable:** `src/predict.py` + `scripts/predict_match.py`

---

## How you’ll run things

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/explore.py
python scripts/prepare_data.py
python scripts/train.py
python scripts/predict_match.py "Arsenal" "Chelsea"
```

---

## What “good” looks like

- Beating the always-home baseline matters more than a fancy algorithm
- ~45–55% accuracy is a realistic range for 3-way football results; 80%+ usually means leakage or a bug
- Draws are hard — expect the model to struggle there at first
