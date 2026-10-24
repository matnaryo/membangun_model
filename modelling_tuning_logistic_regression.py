import pandas as pd
import numpy as np
import os
import json
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

# =========================
# DAGSHUB SETUP
# =========================
USE_MLFLOW = True
USE_DAGSHUB = False

if USE_MLFLOW:
    if USE_DAGSHUB:
        os.environ["MLFLOW_TRACKING_URI"] = (
            "https://dagshub.com/rohmat.sunaryo/Eksperimen_SML_Rohmat-sunaryo.mlflow"
        )
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")  # Lokal tracking

mlflow.set_experiment("Loan Default Tuning")

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(BASE_DIR, "train_processed.csv"))

X = data.drop("Default", axis=1)
y = data["Default"]

# =========================
# SAMPLING (WAJIB untuk laptop)
# =========================
X_small, _, y_small, _ = train_test_split(
    X, y, train_size=0.2, stratify=y, random_state=42
)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2, stratify=y_small, random_state=42
)

# =========================
# SCALING (WAJIB untuk Logistic)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with mlflow.start_run(run_name="Logistic Regression Baseline"):

    # =========================
    # MODEL
    # =========================
    model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)

    print("Mulai training...")
    model.fit(X_train_scaled, y_train)

    # =========================
    # EVALUASI
    # =========================
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    # =========================
    # LOG PARAMETER
    # =========================
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("sampling", "20%")

    # =========================
    # LOG METRIC
    # =========================
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    report = classification_report(y_test, y_pred, output_dict=True)

    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    mlflow.log_artifact("classification_report.json")

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    importance = model.coef_[0]

    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": importance}
    ).sort_values(by="importance", ascending=False)

    feature_importance.to_csv("feature_importance.csv", index=False)

    mlflow.log_artifact("feature_importance.csv")

    # =========================
    # LOG MODEL
    # =========================
    mlflow.sklearn.log_model(model, "model")

    print("Run ID:", mlflow.active_run().info.run_id)
    print("Run Name:", mlflow.active_run().data.tags.get("mlflow.runName"))
