# Prem Predictor

Predict Premier League match results (Home / Draw / Away).

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py
python scripts/build_features.py
python scripts/train.py
python scripts/predict_match.py "Arsenal" "Chelsea"
```
