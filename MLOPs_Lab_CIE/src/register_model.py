import mlflow
import json
import os

client = mlflow.tracking.MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name("launchpredict-countdown-hold-min")

experiment_id = experiment.experiment_id

# Get runs sorted by RMSE
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.rmse ASC"]
)

# Safety check
if len(runs) == 0:
    raise Exception("No MLflow runs found. Run train.py again.")

best_run = runs[0]
run_id = best_run.info.run_id

# IMPORTANT: use correct model name logged earlier
model_uri = f"runs:/{run_id}/LinearRegression"

model_name = "launchpredict-countdown-hold-min-predictor"

result = mlflow.register_model(model_uri, model_name)

output = {
    "registered_model_name": model_name,
    "version": result.version,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": best_run.data.metrics["rmse"]
}

os.makedirs("results", exist_ok=True)

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("STEP 3 DONE")