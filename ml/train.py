<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):

    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    lr_model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, lr_model.predict(X_train))
    print(f"Training Accuracy: {train_acc:.4f}")
    
    return lr_model


def train_decision_tree(X_train, y_train):

    
    dt_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=500,
        min_samples_leaf=200,
        ccp_alpha=0.005,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, dt_model.predict(X_train))
    print(f"Training Accuracy: {train_acc:.4f}")
    
    return dt_model


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)
    
    # logistic regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))
    print(f"Test Accuracy: {lr_test_acc:.4f}")
    save_model(lr_model, os.path.join(models_dir, "logistic_regression.pkl"))
    
<<<<<<< HEAD
    # decision tree
    dt_model = train_decision_tree(X_train, y_train)
    dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))
    print(f"Test Accuracy: {dt_test_acc:.4f}")
    save_model(dt_model, os.path.join(models_dir, "decision_tree.pkl"))
    
    # feature importance check
    feature_names = X_train.columns
    importances = dt_model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp:
        if imp > 0:
            print(f"  {name}: {imp:.4f}")
    
    # comparison
    print(f"Logistic Regression - Test Accuracy: {lr_test_acc:.4f}")
    print(f"Decision Tree       - Test Accuracy: {dt_test_acc:.4f}")

<<<<<<< HEAD
=======
=======
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

def train_models(X_train, y_train, preprocessor):
    """
    Trains Logistic Regression and Decision Tree models using the preprocessor.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    trained_pipelines = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
    return trained_pipelines
>>>>>>> 34144b8 (train the model)
>>>>>>> c868f44e5d3ee6b6cfa53894b7380e6241aba332
=======
>>>>>>> a08bcb3 (add model evaluation and comparison report)
