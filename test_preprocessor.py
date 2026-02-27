"""
test_preprocessor.py — Verify the saved preprocessor is valid.

This script LOADS and VERIFIES the existing preprocessor.pkl.
It does NOT overwrite it. Run `python train.py` to regenerate artifacts.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "maintenance_model.pkl")


def test_preprocessor_exists():
    """Check that the preprocessor file exists and is non-empty."""
    assert os.path.exists(PREPROCESSOR_PATH), (
        f"preprocessor.pkl not found at {PREPROCESSOR_PATH}. Run `python train.py` first."
    )
    size = os.path.getsize(PREPROCESSOR_PATH)
    assert size > 100, f"preprocessor.pkl is too small ({size} bytes) — likely corrupted."
    print(f"✅ preprocessor.pkl exists ({size} bytes)")


def test_preprocessor_is_fitted():
    """Verify the saved preprocessor is fitted (not just built)."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    try:
        check_is_fitted(preprocessor)
        print(f"✅ Preprocessor is fitted (type: {type(preprocessor).__name__})")
    except Exception as e:
        raise AssertionError(f"Preprocessor is NOT fitted: {e}")


def test_transform_produces_correct_shape():
    """Run a sample input through the preprocessor and check output shape."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    sample = pd.DataFrame([{
        "Mileage": 55000,
        "Reported_Issues": 2,
        "Vehicle_Age": 5,
        "Engine_Size": 1500,
        "Odometer_Reading": 75000,
        "Insurance_Premium": 17500,
        "Service_History": 5,
        "Accident_History": 1,
        "Fuel_Efficiency": 15.0,
        "Last_Service_Days_Ago": 900,
        "Warranty_Days_Remaining": -300,
        "Vehicle_Model": "Car",
        "Maintenance_History": "Good",
        "Fuel_Type": "Petrol",
        "Transmission_Type": "Automatic",
        "Owner_Type": "First",
        "Tire_Condition": "New",
        "Brake_Condition": "New",
        "Battery_Status": "New",
    }])

    result = preprocessor.transform(sample)
    assert result.shape[0] == 1, f"Expected 1 row, got {result.shape[0]}"
    assert result.shape[1] > 0, "Output has 0 columns"
    assert not np.isnan(result).any(), "NaN values found in output"
    print(f"✅ Transform produces valid output: shape {result.shape}, no NaNs")


def test_model_compatible():
    """Verify the model can accept preprocessor output."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "Mileage": 55000,
        "Reported_Issues": 2,
        "Vehicle_Age": 5,
        "Engine_Size": 1500,
        "Odometer_Reading": 75000,
        "Insurance_Premium": 17500,
        "Service_History": 5,
        "Accident_History": 1,
        "Fuel_Efficiency": 15.0,
        "Last_Service_Days_Ago": 900,
        "Warranty_Days_Remaining": -300,
        "Vehicle_Model": "Car",
        "Maintenance_History": "Good",
        "Fuel_Type": "Petrol",
        "Transmission_Type": "Automatic",
        "Owner_Type": "First",
        "Tire_Condition": "New",
        "Brake_Condition": "New",
        "Battery_Status": "New",
    }])

    X = preprocessor.transform(sample)
    pred = model.predict(X)
    prob = model.predict_proba(X)

    assert pred[0] in [0, 1], f"Unexpected prediction: {pred[0]}"
    assert 0.0 <= prob[0][1] <= 1.0, f"Probability out of range: {prob[0][1]}"
    print(f"✅ End-to-end prediction works: class={pred[0]}, prob={prob[0][1]:.4f}")


if __name__ == "__main__":
    print("=== Preprocessor & Model Verification ===\n")
    tests = [
        test_preprocessor_exists,
        test_preprocessor_is_fitted,
        test_transform_produces_correct_shape,
        test_model_compatible,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 All checks passed!")
    else:
        print("⚠️  Some checks failed. Run `python train.py` to regenerate artifacts.")
        sys.exit(1)