from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Calculates key classification metrics for a given model.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }
    
    return metrics

def get_feature_importance(model, feature_names):
    """
    Extracts feature importance from a Decision Tree model.
    """
    # Access the classifier from the pipeline
    classifier = model.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        # Note: In a real scenario, we'd need to map these back to the encoded features properly.
        # For simplicity in this structure, we'll return the raw importance array 
        # normally expected after preprocessing.
        return dict(zip(feature_names, importances))
    return None
