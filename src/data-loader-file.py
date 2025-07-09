"""
Data loading and preprocessing utilities for velocity prediction models.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv_files(file_list, data_dir='data'):
    """
    Load and concatenate multiple CSV files.
    
    Parameters:
    -----------
    file_list : list
        List of CSV filenames to load
    data_dir : str
        Directory containing the CSV files
        
    Returns:
    --------
    pandas.DataFrame
        Concatenated dataframe from all CSV files
    """
    dfs = []
    for csv_file in file_list:
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Loaded {csv_file}, shape: {df.shape}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        raise ValueError("No CSV files were successfully loaded")
    
    combined_df = pd.concat(dfs)
    print(f"Combined dataframe shape: {combined_df.shape}")
    return combined_df

def prepare_training_data(combined_df, feature_cols, target_col, test_size, random_state):
    """
    Extract features and target from the dataframe and split into training and testing sets.
    
    Parameters:
    -----------
    combined_df : pandas.DataFrame
        Combined dataframe containing all data
    feature_cols : slice or list
        Columns to use as features
    target_col : int
        Column index for the target variable
    test_size : float
        Proportion of data to reserve for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) - Train-test split of inputs and targets
    """
    features = combined_df.iloc[:, feature_cols]
    target = combined_df.iloc[:, target_col]
    
    print(f"Features shape: {features.shape}, Target shape: {target.shape}")
    
    # Check for NaN values
    if features.isna().any().any() or target.isna().any():
        print("Warning: NaN values found in features or target")
        print(f"Features NaNs: {features.isna().sum().sum()}")
        print(f"Target NaNs: {target.isna().sum()}")
        
        # Drop NaN values
        valid_indices = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_indices]
        target = target[valid_indices]
        print(f"After dropping NaNs - Features shape: {features.shape}, Target shape: {target.shape}")
    
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

def load_test_data(test_file, feature_cols, data_dir='data'):
    """
    Load test data for prediction.
    
    Parameters:
    -----------
    test_file : str
        Name of the test CSV file
    feature_cols : slice or list
        Columns to use as features
    data_dir : str
        Directory containing the test file
        
    Returns:
    --------
    pandas.DataFrame
        Features extracted from the test file
    """
    file_path = os.path.join(data_dir, test_file)
    test_df = pd.read_csv(file_path)
    test_features = test_df.iloc[:, feature_cols]
    
    print(f"Loaded test file {test_file}, features shape: {test_features.shape}")
    
    return test_features
