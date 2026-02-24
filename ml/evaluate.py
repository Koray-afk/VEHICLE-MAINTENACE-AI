<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(data_dir):
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, target_names=['No Maintenance', 'Needs Maintenance'])
    
    result = {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'rmse': rmse,
        'confusion_matrix': cm,
        'report': report
    }
    
    return result


def print_results(result):
    print(f"  {result['model_name']}")
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1 Score:  {result['f1_score']:.4f}")
    print(f"  RMSE:      {result['rmse']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {result['confusion_matrix']}")
    print(f"\n{result['report']}")


def save_report(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("MODEL EVALUATION REPORT\n")
        
        
        for result in results:
            f.write(f"Model: {result['model_name']}\n")
        
            f.write(f"Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall:    {result['recall']:.4f}\n")
            f.write(f"F1 Score:  {result['f1_score']:.4f}\n")
            f.write(f"RMSE:      {result['rmse']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(result['confusion_matrix']) + "\n\n")
            f.write("Classification Report:\n")
            f.write(result['report'] + "\n")
        
        
       
        f.write("\nCOMPARISON SUMMARY\n")
     
        f.write(f"{'Metric':<15} {'Logistic Reg':<15} {'Decision Tree':<15}\n")
        
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'rmse']:
            val1 = results[0][metric]
            val2 = results[1][metric]
            f.write(f"{metric:<15} {val1:<15.4f} {val2:<15.4f}\n")
    



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    
    X_test, y_test = load_processed_data(data_dir)
    
    # load models
    lr_model = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
    dt_model = joblib.load(os.path.join(models_dir, "decision_tree.pkl"))
    
    # evaluate
    lr_result = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    dt_result = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    
    print_results(lr_result)
    print_results(dt_result)
    
    # save report
    report_path = os.path.join(reports_dir, "model_comparison.txt")
    save_report([lr_result, dt_result], report_path)
    
    
=======
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
>>>>>>> c868f44e5d3ee6b6cfa53894b7380e6241aba332
