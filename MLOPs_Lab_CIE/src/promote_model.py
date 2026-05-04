import mlflow
import json
import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("data/training_data.csv")

X = df.drop("countdown_hold_min", axis=1)
y = df["countdown_hold_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

model = RandomForestRegressor(random_state=99)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse_new = np.sqrt(mean_squared_error(y_test, preds))

client = mlflow.tracking.MlflowClient()

name = "launchpredict-countdown-hold-min-predictor"

# Get version 1
versions = client.search_model_versions(f"name='{name}'")

v1 = int(versions[0].version)

# Register new version
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    run_id = mlflow.active_run().info.run_id

model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, name)

v2 = int(result.version)

# Compare
action = "kept"
champion = v1

if rmse_new < 100:  # simple condition
    client.set_registered_model_alias(name, "champion", v2)
    action = "promoted"
    champion = v2
else:
    client.set_registered_model_alias(name, "champion", v1)

output = {
    "registered_model_name": name,
    "alias_name": "champion",
    "champion_version": champion,
    "challenger_version": v2,
    "action": action
}

os.makedirs("results", exist_ok=True)

with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print("STEP 4 DONE")