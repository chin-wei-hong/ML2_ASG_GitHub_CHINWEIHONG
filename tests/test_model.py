import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MODEL_PATH = "models/selected_model.joblib"
DATA_PATH = "data/day_2011.csv"

RMSE_BASELINE = 648.015
IMPROVEMENT_FACTOR = 0.95

def main():
    df = pd.read_csv(DATA_PATH)

    # date features (only if you used them in training)
    df["dteday"] = pd.to_datetime(df["dteday"], dayfirst=True)
    df["day"] = df["dteday"].dt.day
    df["year"] = df["dteday"].dt.year

    y = df["cnt"]
    X = df.drop(columns=["cnt", "dteday"], errors="ignore")

    # MATCH YOUR TRAINING OHE EXACTLY
    X = pd.get_dummies(
        X,
        columns=["season", "weathersit", "weekday", "mnth"],
        drop_first=True,
        dtype=int
    )

    # split (same as your work)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)

    # Align columns to what the model was trained on (important!)
    trained_cols = list(model.feature_names_in_)
    X_test = X_test.reindex(columns=trained_cols, fill_value=0)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    threshold = IMPROVEMENT_FACTOR * RMSE_BASELINE
    print(f"RMSE(model)     = {rmse:.3f}")
    print(f"RMSE(baseline)  = {RMSE_BASELINE:.3f}")
    print(f"RMSE(threshold) = {threshold:.3f}")

    if rmse > threshold:
        print("FAILED QUALITY GATE ❌")
        sys.exit(1)

    print("PASSED QUALITY GATE ✅")
    sys.exit(0)

if __name__ == "__main__":
    main()
