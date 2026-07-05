"""
Re-runs the model-persistence step from train_and_evaluate.py on demand,
so the real-time pipeline can pick up a freshly refit model.

Honest limitation: a bounded demo session doesn't generate genuinely new
training labels. A CAR needs several trading days of subsequent price
history before it's a valid label, so nothing filed during a short demo
session has matured into something to learn from yet. This script
demonstrates the retraining mechanism against the existing historical
dataset (the same features/target that were already there), not real
online learning from the demo's own data. Re-run
build_features.py first if the underlying historical data has changed.

Usage:
    python retrain_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "04_predictive_model"))
import config
import joblib
import pandas as pd
from train_and_evaluate import ALL_FEATURES, get_models


def main():
    df = pd.read_parquet(config.MODEL_FEATURES_PATH)

    pipeline = get_models()["Ridge"]
    pipeline.fit(df[ALL_FEATURES], df["car_primary"])
    joblib.dump(pipeline, config.MODEL_PIPELINE_PATH)

    print(f"Refit Ridge on {len(df)} rows from {config.MODEL_FEATURES_PATH}, "
          f"saved to {config.MODEL_PIPELINE_PATH}.")
    print("This demonstrates the retraining mechanism against the existing historical")
    print("dataset. A short demo session doesn't generate new matured labels (a CAR")
    print("needs several trading days of subsequent price history first), so this is")
    print("not exercising real online learning from live demo data.")


if __name__ == "__main__":
    main()
