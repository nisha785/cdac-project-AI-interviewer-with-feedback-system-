"""
aggregator.py

This module calculates the overall scores and generates feedback for each candidate's response.
It reads the consolidated scores, computes a weighted average, and writes the results out.
It includes validation and exception handling for production readiness.
"""
import pandas as pd
import sys
import os

def generate_feedback(score):
    """
    Generates textual feedback based on the overall score.
    """
    if pd.isna(score):
        return "No score available"
    if score >= 80:
        return "Excellent performance"
    elif score >= 60:
        return "Good performance"
    else:
        return "Needs more practice"

def aggregate_scores(input_file, output_file):
    """
    Reads the input CSV, calculates the overall score, generates feedback, 
    and saves the output. Includes checks for missing columns and invalid data.
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    required_columns = ["accuracy_score", "speech_score", "facial_score"]
    
    # Check if required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns for calculation: {', '.join(missing_cols)}")
        sys.exit(1)

    # Handle missing data (NaN) by filling with 0 or dropping. 
    # Here we drop rows where any of the required scores are NaN to prevent invalid calculations.
    initial_len = len(df)
    df.dropna(subset=required_columns, inplace=True)
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} row(s) due to missing score data.")

    # Calculate overall score safely
    try:
        df["overall_score"] = (
            0.5 * pd.to_numeric(df["accuracy_score"], errors='coerce') +
            0.25 * pd.to_numeric(df["speech_score"], errors='coerce') +
            0.25 * pd.to_numeric(df["facial_score"], errors='coerce')
        )
    except Exception as e:
        print(f"Error calculating overall score (invalid values present): {e}")
        sys.exit(1)

    # Handle any NaN created during numeric conversion
    df.dropna(subset=["overall_score"], inplace=True)

    # Generate feedback
    df["feedback"] = df["overall_score"].apply(generate_feedback)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save final output
    try:
        df.to_csv(output_file, index=False)
        print(f"Aggregation successful. Output saved to {output_file}")
        print("\nPreview of processed data:")
        print(df.head())
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Integration Readiness: You can change these paths when real data arrives
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.path.abspath(os.path.join(os.getcwd(), "analytics"))
        if not os.path.exists(os.path.join(script_dir, "sample_data")):
            script_dir = os.getcwd()
            
    INPUT_PATH = os.path.join(script_dir, "sample_data", "mock_scores.csv")
    OUTPUT_PATH = os.path.join(script_dir, "sample_data", "final_scores.csv")
    
    aggregate_scores(INPUT_PATH, OUTPUT_PATH)