"""
Dataset Validation Script
Validates minimum requirements: 12+ features and 500+ instances
"""

import pandas as pd
import os

def validate_dataset():
    """Validate dataset meets minimum requirements"""
    
    # Get data directory path
    data_dir = os.path.dirname(__file__)
    if not os.path.exists(data_dir):
        data_dir = 'data'
    
    # Find CSV files in data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ ERROR: No CSV files found in data/ folder")
        return False
    
    # Use the first CSV file found
    dataset_file = csv_files[0]
    file_path = os.path.join(data_dir, dataset_file)
    
    print(f"Loading dataset: {dataset_file}")
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Get dimensions
        n_instances = len(df)
        n_features = len(df.columns) - 1  # Exclude target column
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Instances: {n_instances}")
        print(f"   Features: {n_features}")
        
        # Check minimum requirements
        print(f"\n📋 Checking Requirements:")
        
        # Check minimum instances (500+)
        if n_instances >= 500:
            print(f"✅ Instances: {n_instances} (meets requirement of 500+)")
            instances_ok = True
        else:
            print(f"❌ Instances: {n_instances} (minimum required: 500)")
            instances_ok = False
        
        # Check minimum features (12+)
        if n_features >= 12:
            print(f"✅ Features: {n_features} (meets requirement of 12+)")
            features_ok = True
        else:
            print(f"❌ Features: {n_features} (minimum required: 12)")
            features_ok = False
        
        # Final result
        if instances_ok and features_ok:
            print(f"\n✅ Dataset meets all requirements!")
            return True
        else:
            print(f"\n❌ Dataset does not meet requirements!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Could not load dataset: {e}")
        return False

if __name__ == "__main__":
    validate_dataset()
