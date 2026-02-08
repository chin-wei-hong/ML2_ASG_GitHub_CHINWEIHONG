# ML2 Assignment 1 - Task 3 (GitHub Actions Quality Gate)

## Repo structure
- src/: Task 1 & 2 notebook/script
- tests/: test_model.py quality gate
- data/: evaluation dataset (day_2011.csv)
- models/: saved best model (selected_model.joblib)
- .github/workflows/: GitHub Actions workflow

## Run locally
pip install -r requirements.txt
python tests/test_model.py
