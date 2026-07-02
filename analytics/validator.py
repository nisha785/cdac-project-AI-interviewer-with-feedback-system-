"""
validator.py

This module is responsible for validating the incoming mock scores data.
It ensures that all required columns exist, there are no missing values,
and that all score values fall within the valid range of 0 to 100.
"""
import pandas as pd
import sys
import os

# Define the columns that must be present in the dataset
REQUIRED_COLUMNS = [
    "session_id",
    "question_id",
    "accuracy_score",
    "speech_score",
    "facial_score"
]

def validate_data(file_path):
    """
    Reads a CSV file and validates its contents against specific rules.
    
    Args:
        file_path (str): The path to the CSV file to validate.
        
    Returns:
        bool: True if validation is successful, False otherwise.
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return False
    except Exception as e:
        print(f"Error loading file: {e}")
        return False

    is_valid = True

    # 1. Verify required columns exist
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        print(f"Missing column(s): {', '.join(missing_columns)}")
        is_valid = False

    # 2. Verify no missing values
    if df.isnull().values.any():
        print("Error: Dataset contains missing (null) values.")
        is_valid = False

    # 3. Verify all scores lie between 0 and 100
    score_columns = ["accuracy_score", "speech_score", "facial_score"]
    for col in score_columns:
        if col in df.columns:
            invalid_scores = df[(pd.to_numeric(df[col], errors='coerce') < 0) | (pd.to_numeric(df[col], errors='coerce') > 100)]
            if not invalid_scores.empty:
                print(f"Invalid score detected in {col}. Scores must be between 0 and 100.")
                is_valid = False

    if is_valid:
        print("Validation Successful")
        
    return is_valid

if __name__ == "__main__":
    # Path to the mock scores file relative to this script
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.path.abspath(os.path.join(os.getcwd(), "analytics"))
        if not os.path.exists(os.path.join(script_dir, "sample_data")):
            script_dir = os.getcwd()
            
    data_file = os.path.join(script_dir, "sample_data", "mock_scores.csv")
    validate_data(data_file)