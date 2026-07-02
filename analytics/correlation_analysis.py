"""
correlation_analysis.py

This module computes the correlation matrix between the different score metrics.
Correlation helps us understand if and how strongly pairs of scores are related.
The output is saved as a CSV.
"""
import pandas as pd
import os

def perform_correlation_analysis(file_path, output_path):
    """
    Calculates the correlation matrix for numerical score columns and saves it.
    
    Args:
        file_path (str): The path to the input CSV file.
        output_path (str): The path where the correlation matrix CSV will be saved.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    target_columns = ["accuracy_score", "speech_score", "facial_score"]
    if "overall_score" in df.columns:
        target_columns.append("overall_score")
        
    available_cols = [col for col in target_columns if col in df.columns]
    
    # Convert to numeric to ensure correlation works
    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Compute the correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    corr_matrix.to_csv(output_path)
    print(f"Correlation matrix saved successfully to {output_path}")
    print("\n--- Correlation Explanation ---")
    print("Values close to 1 indicate a strong positive relationship.")
    print("Values close to -1 indicate a strong negative relationship.")
    print("Values close to 0 indicate little or no relationship.")

if __name__ == "__main__":
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.path.abspath(os.path.join(os.getcwd(), "analytics"))
        if not os.path.exists(os.path.join(script_dir, "sample_data")):
            script_dir = os.getcwd()
            
    input_file = os.path.join(script_dir, "sample_data", "final_scores.csv")
    if not os.path.exists(input_file):
        input_file = os.path.join(script_dir, "sample_data", "mock_scores.csv")
        
    output_file = os.path.join(script_dir, "sample_data", "correlation_matrix.csv")
    perform_correlation_analysis(input_file, output_file)
