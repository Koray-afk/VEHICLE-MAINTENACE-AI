"""
train.py — Vehicle Maintenance AI
==================================
Pipeline:
  • Load raw CSV
  • Engineer date features
  • Train/test split  (NO preprocessing leakage)
  • Fit preprocessor on TRAINING data only
  • Train Logistic Regression + Decision Tree (regularized)
  • Log evaluation metrics
  • Save artifacts to models/

Usage:
    python train.py
"""

import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
import joblib

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_CSV       = os.path.join(BASE_DIR, "data", "raw", "vehicle_maintenance_data.csv")
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "vehicle_maintenance_cleaned.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Column definitions (derived from the actual dataset schema)
# ---------------------------------------------------------------------------
TARGET_COL = "Need_Maintenance"

# Date columns — extracted to numeric features, then dropped
DATE_COLS = ["Last_Service_Date", "Warranty_Expiry_Date"]

# Numerical features (after date extraction)
NUMERICAL_COLS = [
    "Mileage",
    "Reported_Issues",
    "Vehicle_Age",
    "Engine_Size",
    "Odometer_Reading",
    "Insurance_Premium",
    "Service_History",
    "Accident_History",
    "Fuel_Efficiency",
    "Last_Service_Days_Ago",      # derived from Last_Service_Date
    "Warranty_Days_Remaining",    # derived from Warranty_Expiry_Date
]

# Categorical features
CATEGORICAL_COLS = [
    "Vehicle_Model",
    "Maintenance_History",
    "Fuel_Type",
    "Transmission_Type",
    "Owner_Type",
    "Tire_Condition",
    "Brake_Condition",
    "Battery_Status",
]


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading raw data from: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Feature engineering on date columns
# ---------------------------------------------------------------------------
def engineer_dates(df: pd.DataFrame, reference_date: str = "2026-02-26") -> pd.DataFrame:
    """Convert date strings to useful numeric features, then drop originals."""
    ref = pd.Timestamp(reference_date)

    df["Last_Service_Date"]    = pd.to_datetime(df["Last_Service_Date"],    errors="coerce")
    df["Warranty_Expiry_Date"] = pd.to_datetime(df["Warranty_Expiry_Date"], errors="coerce")

    df["Last_Service_Days_Ago"]   = (ref - df["Last_Service_Date"]).dt.days
    df["Warranty_Days_Remaining"] = (df["Warranty_Expiry_Date"] - ref).dt.days

    df.drop(columns=DATE_COLS, inplace=True)
    logger.info("Date columns converted to numeric features.")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Build sklearn ColumnTransformer pipeline
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer with:
      • Numerical branch : median imputation → StandardScaler
      • Categorical branch: most-frequent imputation → OneHotEncoder
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline,  NUMERICAL_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ], remainder="drop")

    return preprocessor


# ---------------------------------------------------------------------------
# Step 4 — Build feature-name list from a fitted preprocessor
# ---------------------------------------------------------------------------
def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return the full list of output column names from a fitted preprocessor."""
    ohe_feature_names = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    return NUMERICAL_COLS + ohe_feature_names


# ---------------------------------------------------------------------------
# Step 5 — Evaluate and log model metrics
# ---------------------------------------------------------------------------
def evaluate_model(name: str, model, X_train, y_train, X_test, y_test):
    """Compute and log key metrics for a trained model."""
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc  = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred)
    recall    = recall_score(y_test, test_pred)
    f1        = f1_score(y_test, test_pred)

    logger.info("--- %s Metrics ---", name)
    logger.info("  Train Accuracy : %.4f", train_acc)
    logger.info("  Test  Accuracy : %.4f", test_acc)
    logger.info("  Precision      : %.4f", precision)
    logger.info("  Recall         : %.4f", recall)
    logger.info("  F1 Score       : %.4f", f1)
    logger.info("  Train-Test Gap : %.4f", train_acc - test_acc)

    if hasattr(model, "get_depth"):
        logger.info("  Tree Depth     : %d", model.get_depth())
        logger.info("  Tree Leaves    : %d", model.get_n_leaves())

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Step 6 — Train models (split → fit preprocessor → train → evaluate → save)
# ---------------------------------------------------------------------------
def train_and_save(df: pd.DataFrame):
    """
    Correct pipeline: split FIRST, then fit preprocessor on training data only.
    This eliminates data leakage from test set into scaler statistics.
    """
    # Separate features and target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # --- Split FIRST (before any preprocessing) ---
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        "Train/test split: %d train, %d test",
        len(X_train_raw), len(X_test_raw),
    )

    # --- Fit preprocessor on TRAINING data only ---
    logger.info("Building preprocessing pipeline …")
    preprocessor = build_preprocessor()

    logger.info("Fitting preprocessor on TRAINING data only …")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    # Save the fitted preprocessor
    os.makedirs(MODELS_DIR, exist_ok=True)
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Preprocessor saved to: %s", preprocessor_path)

    # --- Save processed data (for EDA notebook) ---
    feature_names = get_feature_names(preprocessor)
    all_transformed = preprocessor.transform(X)  # transform full data for CSV
    cleaned_df = pd.DataFrame(all_transformed, columns=feature_names)
    cleaned_df[TARGET_COL] = y.values
    os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
    cleaned_df.to_csv(PROCESSED_CSV, index=False)
    logger.info("Cleaned data saved to: %s", PROCESSED_CSV)

    # ---------------------------------------------------------------
    # Train Logistic Regression
    # ---------------------------------------------------------------
    logger.info("Training Logistic Regression …")
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(
        "Logistic Regression", lr_model, X_train, y_train, X_test, y_test
    )
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_model.pkl"))

    # ---------------------------------------------------------------
    # Train Decision Tree (regularized to prevent overfitting)
    # ---------------------------------------------------------------
    logger.info("Training Decision Tree (regularized) …")
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
    )
    dt_model.fit(X_train, y_train)
    dt_metrics = evaluate_model(
        "Decision Tree", dt_model, X_train, y_train, X_test, y_test
    )
    joblib.dump(dt_model, os.path.join(MODELS_DIR, "decision_tree_model.pkl"))

    # ---------------------------------------------------------------
    # Deploy Logistic Regression as the main model
    # ---------------------------------------------------------------
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "maintenance_model.pkl"))
    logger.info("Deployed model (Logistic Regression) saved as maintenance_model.pkl")

    # ---------------------------------------------------------------
    # Save training metadata
    # ---------------------------------------------------------------
    import json
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "deployed_model": "LogisticRegression",
        "sklearn_version": __import__("sklearn").__version__,
        "logistic_regression": lr_metrics,
        "decision_tree": dt_metrics,
    }
    metadata_path = os.path.join(MODELS_DIR, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Training metadata saved to: %s", metadata_path)

    return lr_model, dt_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=== Vehicle Maintenance AI — Training Pipeline ===")

    df = load_data(RAW_CSV)
    df = engineer_dates(df)

    lr_model, dt_model = train_and_save(df)

    logger.info("=== Training complete! ===")


if __name__ == "__main__":
    main()
