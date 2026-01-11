"""
Script to generate Excel file with all model results
This creates a comprehensive Excel file with all evaluation metrics
"""

import pandas as pd
import os
import sys

# Add parent directory to path to import train_models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.train_models import main as train_models

def generate_results_excel():
    """Generate Excel file with all model results"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'model_results.xlsx')
    output_path = os.path.join(script_dir, 'ML_Assignment_2_Results.xlsx')
    
    # Check if models are already trained
    if not os.path.exists(results_path):
        print("Models not trained yet. Training models first...")
        train_models()
    
    # Read existing results
    results_df = pd.read_excel(results_path, index_col=0)
    
    # Create a more detailed Excel file with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary Table
        summary_df = results_df.copy()
        summary_df.columns = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        summary_df.to_excel(writer, sheet_name='Model Comparison', index=True)
        
        # Sheet 2: Detailed Metrics
        detailed_df = results_df.T
        detailed_df.to_excel(writer, sheet_name='Detailed Metrics', index=True)
        
        # Sheet 3: Best Model per Metric
        best_models = pd.DataFrame({
            'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            'Best Model': [
                results_df['Accuracy'].idxmax(),
                results_df['AUC'].idxmax(),
                results_df['Precision'].idxmax(),
                results_df['Recall'].idxmax(),
                results_df['F1'].idxmax(),
                results_df['MCC'].idxmax()
            ],
            'Best Value': [
                results_df['Accuracy'].max(),
                results_df['AUC'].max(),
                results_df['Precision'].max(),
                results_df['Recall'].max(),
                results_df['F1'].max(),
                results_df['MCC'].max()
            ]
        })
        best_models.to_excel(writer, sheet_name='Best Models', index=False)
        
        # Sheet 4: Model Rankings
        rankings = pd.DataFrame(index=results_df.index)
        for metric in results_df.columns:
            rankings[f'{metric}_Rank'] = results_df[metric].rank(ascending=False, method='min').astype(int)
        rankings['Average_Rank'] = rankings.mean(axis=1).round(2)
        rankings = rankings.sort_values('Average_Rank')
        rankings.to_excel(writer, sheet_name='Model Rankings', index=True)
    
    print("\n" + "="*50)
    print(f"✅ Excel file generated: {output_path}")
    print("="*50)
    print("\nSheets created:")
    print("1. Model Comparison - Summary table with all metrics")
    print("2. Detailed Metrics - Transposed view of metrics")
    print("3. Best Models - Best performing model for each metric")
    print("4. Model Rankings - Ranking of models by each metric")
    print("\nResults Summary:")
    print(results_df.round(4))

if __name__ == "__main__":
    generate_results_excel()

