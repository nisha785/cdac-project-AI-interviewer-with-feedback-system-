"""
visualization.py

This module generates visual representations of the scores, including:
- Histograms of individual scores
- A box plot for score distributions
- A bar chart for average scores
- A distribution plot for the overall scores
All generated plots are saved into the analytics/plots/ directory.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_visualizations(file_path, output_dir):
    """
    Generates and saves several plots based on the score data.
    
    Args:
        file_path (str): The path to the input CSV file.
        output_dir (str): The directory where the plots will be saved.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    score_cols = ["accuracy_score", "speech_score", "facial_score"]
    
    # Convert available scores to numeric
    for col in score_cols + (["overall_score"] if "overall_score" in df.columns else []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Histogram of individual scores
    plt.figure(figsize=(10, 6))
    for col in score_cols:
        if col in df.columns:
            plt.hist(df[col].dropna(), bins=10, alpha=0.5, label=col)
    plt.title("Histogram of Component Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(output_dir, "histogram_scores.png"))
    plt.close()
    
    # 2. Box Plot of scores
    plt.figure(figsize=(8, 6))
    available_cols = [col for col in score_cols if col in df.columns]
    if available_cols:
        df.boxplot(column=available_cols)
        plt.title("Box Plot of Component Scores")
        plt.ylabel("Score")
        plt.savefig(os.path.join(output_dir, "boxplot_scores.png"))
    plt.close()
    
    # 3. Bar Chart of average scores
    plt.figure(figsize=(8, 6))
    if available_cols:
        avg_scores = df[available_cols].mean()
        avg_scores.plot(kind='bar')
        plt.title("Average Scores by Category")
        plt.ylabel("Average Score")
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(output_dir, "barchart_average_scores.png"))
    plt.close()
    
    # 4. Distribution of overall scores (if available)
    if "overall_score" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df["overall_score"].dropna(), bins=10, color='purple', alpha=0.7, edgecolor='black')
        plt.title("Distribution of Overall Scores")
        plt.xlabel("Overall Score")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(os.path.join(output_dir, "distribution_overall_scores.png"))
        plt.close()
        
    print(f"All visualizations have been successfully saved to '{output_dir}'.")

if __name__ == "__main__":
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        # Fallback if you run this in an interactive window (like Jupyter/VSCode selection)
        script_dir = os.path.abspath(os.path.join(os.getcwd(), "analytics"))
        if not os.path.exists(os.path.join(script_dir, "sample_data")):
            script_dir = os.getcwd()

    input_file = os.path.join(script_dir, "sample_data", "final_scores.csv")
    if not os.path.exists(input_file):
        input_file = os.path.join(script_dir, "sample_data", "mock_scores.csv")
        
    plots_dir = os.path.join(script_dir, "plots")
    generate_visualizations(input_file, plots_dir)
