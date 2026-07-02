"""
statistics_report.py

This module calculates various statistical measures (mean, median, mode, 
standard deviation, variance, min, and max) for the different score categories.
It saves the results into a CSV file for reporting.
"""
import pandas as pd
import os

def calculate_statistics(file_path, output_path):
    """
    Calculates statistics for score columns and saves the summary.
    
    Args:
        file_path (str): The path to the input CSV file.
        output_path (str): The path where the summary CSV will be saved.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run aggregator.py first if needed.")
        return

    target_columns = ["accuracy_score", "speech_score", "facial_score"]
    if "overall_score" in df.columns:
        target_columns.append("overall_score")
        
    # Filter only available columns to prevent KeyError
    available_cols = [col for col in target_columns if col in df.columns]
    
    stats_list = []
    
    for col in available_cols:
        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
        if not col_data.empty:
            stats = {
                "Metric": col,
                "Mean": col_data.mean(),
                "Median": col_data.median(),
                "Mode": col_data.mode().iloc[0] if not col_data.mode().empty else None,
                "Std_Dev": col_data.std(),
                "Variance": col_data.var(),
                "Min": col_data.min(),
                "Max": col_data.max()
            }
            stats_list.append(stats)
            
    if not stats_list:
        print("No numeric data available to calculate statistics.")
        return

    stats_df = pd.DataFrame(stats_list)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    stats_df.to_csv(output_path, index=False)
    print(f"Statistical summary saved successfully to {output_path}")

if __name__ == "__main__":
    # We use final_scores.csv if available to include overall_score, else mock_scores.csv
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.path.abspath(os.path.join(os.getcwd(), "analytics"))
        if not os.path.exists(os.path.join(script_dir, "sample_data")):
            script_dir = os.getcwd()
            
    input_file = os.path.join(script_dir, "sample_data", "final_scores.csv")
    if not os.path.exists(input_file):
        input_file = os.path.join(script_dir, "sample_data", "mock_scores.csv")
        print(f"Note: {input_file} will be used.")
        
    output_file = os.path.join(script_dir, "sample_data", "statistics_summary.csv")
    calculate_statistics(input_file, output_file)
