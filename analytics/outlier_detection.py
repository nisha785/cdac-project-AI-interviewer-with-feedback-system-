"""
outlier_detection.py

This module uses the Interquartile Range (IQR) method to detect outliers
in the overall_score column. Outliers are abnormally high or low scores
compared to the rest of the dataset.
"""
import pandas as pd
import os

def detect_outliers_iqr(file_path, output_path, column_name="overall_score"):
    """
    Identifies outliers in a specified column using the IQR method.
    
    Args:
        file_path (str): The path to the input CSV file.
        output_path (str): The path where the outliers CSV will be saved.
        column_name (str): The column to check for outliers.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run aggregator.py first.")
        return

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the dataset.")
        return

    # Convert to numeric just in case
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    
    # Calculate Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the dataset for outliers
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    outliers.to_csv(output_path, index=False)
    
    print(f"Outlier detection completed. {len(outliers)} outlier(s) found.")
    print(f"Outliers saved successfully to {output_path}")

if __name__ == "__main__":
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.path.abspath(os.path.join(os.getcwd(), "analytics"))
        if not os.path.exists(os.path.join(script_dir, "sample_data")):
            script_dir = os.getcwd()
            
    input_file = os.path.join(script_dir, "sample_data", "final_scores.csv")
    output_file = os.path.join(script_dir, "sample_data", "outliers.csv")
    detect_outliers_iqr(input_file, output_file)
