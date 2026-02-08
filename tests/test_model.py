import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MODEL_PATH = "models/selected_model.joblib"
DATA_PATH = "data/day_2011.csv"

# Baseline LR RMSE from your Task 1 (2011 test)
RMSE_BASELINE = 531.074
IMPROVEMENT_FACTOR = 0.95  # must be <= 0.95 * baseline

def main():
    df = pd.read_csv(DATA_PATH)

    y = df["cnt"]
    X = df.drop(columns=["cnt", "dteday"], errors="ignore")

    # One-hot encoding like your Task 1 setup
    cat_cols = ["season", "mnth", "weekday", "weathersit", "holiday", "workingday"]
    X = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    threshold = IMPROVEMENT_FACTOR * RMSE_BASELINE

    print(f"RMSE(model)     = {rmse:.3f}")
    print(f"RMSE(threshold) = {threshold:.3f}")

    # Quality gate (fail CI if not met)
    if rmse > threshold:
        print("FAILED QUALITY GATE ❌")
        sys.exit(1)

    print("PASSED QUALITY GATE ✅")
    sys.exit(0)

if __name__ == "__main__":
    main()
