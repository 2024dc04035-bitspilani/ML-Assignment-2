"""
Python Script for XGBoost Classifier
"""

from xgboost import XGBClassifier
import os
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
# Create model directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
saved_models_dir = os.path.join(project_root, 'saved_models')
os.makedirs(saved_models_dir, exist_ok=True)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics for multiclass classification"""
    # Ensure y_pred_proba is 2D (n_samples, n_classes) for multiclass
    if y_pred_proba.ndim == 1:
        # If 1D, it means binary - convert to 2D
        y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
    
    # For multiclass classification
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost Classifier"""
    print("Training XGBoost...")
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Keep full 2D array for multiclass
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    model_path = os.path.join(saved_models_dir, 'xgboost.pkl')
    joblib.dump(model, model_path)
    
    return model, metrics
