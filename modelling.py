import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Loan Default")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "")

data = pd.read_csv(os.path.join(MODEL_DIR, "train_processed.csv"))


# Subsampling agar ringan
# Drop target label dari X
X = data.drop("Default", axis=1)
y = data["Default"]

# Set stratify supaya distribusi label tetap sama
X_small, _, y_small, _ = train_test_split(
    X, y, train_size=0.9, stratify=y, random_state=42
)


# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2, stratify=y_small, random_state=42
)

# Input untuk MLflow logging
input_example = X_train.iloc[0:5]

# Train Random Forest ringan
n_estimators = 150
max_depth = 15

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=10,
    class_weight="balanced",
    n_jobs=1,
    random_state=42,
)


# MLflow logging
with mlflow.start_run():
    mlflow.autolog()

    # Train
    model.fit(X_train, y_train)

    # Hitung akurasi
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Simpan model
    mlflow.sklearn.log_model(
        sk_model=model, artifact_path="model", input_example=input_example
    )

print(f"Training selesai. Accuracy: {accuracy:.4f}")
