"""
Model Training Script
Trains all 6 classification models and saves them along with evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from XGBoost_Classifier import train_xgboost
from Random_Forest_Classifier import train_random_forest
from logistic_regression import train_logistic_regression
from Decision_Tree_Classifier import train_decision_tree
from KNeighbors_Classifier  import train_knn
from Naive_Bayes_Classifier import train_naive_bayes
import joblib
import os

# Create model directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
saved_models_dir = os.path.join(project_root, 'saved_models')
os.makedirs(saved_models_dir, exist_ok=True)

def load_and_prepare_data(file_path='/home/cloud/Desktop/ML Assignment 2/data/Student_performance_data.csv'):
    """
    Load and prepare the dataset for training
    """
    # Try different possible paths
    possible_paths = [
        file_path,
        'data/Student_Performance_data.csv',
        '../data/Student_Performance_data.csv'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError(
            f"Dataset not found. \n"
            f"Tried paths: {possible_paths}"
        )
    
    # Separate features and target
    X = df.drop('GradeClass', axis=1)
    y = df['GradeClass']
    
    # Note: Categorical features are already encoded in the processed dataset
    # If there are still categorical features, encode them
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"Encoding remaining categorical features: {categorical_cols}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (important for Logistic Regression, KNN, and Naive Bayes)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def main():
    """Main function to train all models"""
    print("Loading and preparing data...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_prepare_data()
    
    # Save scaler for later use
    scaler_path = os.path.join(saved_models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Train all models
    results = {}
    
    lr_model, lr_metrics = train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
    results['Logistic Regression'] = lr_metrics
    
    dt_model, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)
    results['Decision Tree'] = dt_metrics
    
    knn_model, knn_metrics = train_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    results['kNN'] = knn_metrics
    
    nb_model, nb_metrics = train_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test)
    results['Naive Bayes'] = nb_metrics
    
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    results['Random Forest'] = rf_metrics
    
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    results['XGBoost'] = xgb_metrics
    
    # Save results to Excel
    results_df = pd.DataFrame(results).T
    results_df.columns = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    results_path = os.path.join(project_root, 'model_results.xlsx')
    results_df.to_excel(results_path, index=True)
    
    print("\n" + "="*50)
    print(f"Training Complete! Results saved to {results_path}")
    print("="*50)
    print("\nResults Summary:")
    print(results_df.round(4))
    
    return results

if __name__ == "__main__":
    main()

