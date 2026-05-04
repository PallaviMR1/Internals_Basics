import pandas as pd
import numpy as np
import mlflow
import json
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("countdown_hold_min", axis=1)
y = df["countdown_hold_min"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid (as per question)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

model = RandomForestRegressor(random_state=42)

mlflow.set_experiment("launchpredict-countdown-hold-min")

with mlflow.start_run(run_name="tuning-launchpredict") as parent_run:

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=5,
        cv=5,
        scoring="neg_root_mean_squared_error",  # ✅ CORRECT
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Predictions
    preds = best_model.predict(X_test)

    # ✅ CORRECT RMSE calculation
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # CV RMSE (convert from negative)
    cv_rmse = -search.best_score_

    output = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": 5,
        "best_params": search.best_params_,
        "best_mae": float(rmse),        # (name kept same as template)
        "best_cv_mae": float(cv_rmse),  # (name kept same as template)
        "parent_run_name": "tuning-launchpredict"
    }

    os.makedirs("results", exist_ok=True)

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("STEP 2 DONE")