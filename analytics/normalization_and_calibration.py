"""
normalization_and_calibration.py

This module implements Week 3 tasks for Member 5:
1. Generates a robust mock dataset simulating 5 distinct mock interview sessions, each with 5 questions.
2. Normalizes scores (Accuracy, Speech, Facial) to a 0.0 - 1.0 range.
3. Computes Pearson correlation matrix to study variable interactions.
4. Performs outlier detection using IQR.
5. Saves results to sample_data/normalized_scores.csv.
"""
import pandas as pd
import numpy as np
import os

def generate_multi_session_mock_data():
    """
    Generates mock scores representing 5 mock interviews (session_id 1 to 5).
    Each session consists of 5 technical and HR questions (question_id 1 to 5).
    """
    np.random.seed(42)  # For reproducible mock scores
    
    data = []
    for session_id in range(1, 6):
        # Generate typical values that might correlate slightly
        base_accuracy = np.random.randint(45, 95)
        base_speech = np.random.randint(40, 90)
        base_facial = np.random.randint(50, 95)
        
        for question_id in range(1, 6):
            accuracy_score = np.clip(base_accuracy + np.random.randint(-15, 15), 0, 100)
            speech_score = np.clip(base_speech + np.random.randint(-15, 15), 0, 100)
            facial_score = np.clip(base_facial + np.random.randint(-15, 15), 0, 100)
            
            # Inject a manual outlier in session 3, question 2 to test outlier detection
            if session_id == 3 and question_id == 2:
                accuracy_score = 12  # Unusually low score
                speech_score = 15
                facial_score = 10
                
            data.append({
                "session_id": session_id,
                "question_id": question_id,
                "accuracy_score": float(accuracy_score),
                "speech_score": float(speech_score),
                "facial_score": float(facial_score)
            })
            
    return pd.DataFrame(data)

def normalize_scores(df):
    """
    Normalizes score columns from 0-100 scale to 0.0-1.0 scale.
    """
    norm_df = df.copy()
    score_cols = ["accuracy_score", "speech_score", "facial_score"]
    
    for col in score_cols:
        norm_df[col] = norm_df[col] / 100.0
        
    return norm_df

def compute_overall_score(df):
    """
    Calculates the overall score using normalized weights:
    overall_score = 0.5 * accuracy + 0.25 * speech + 0.25 * facial
    """
    df["overall_score"] = (
        0.5 * df["accuracy_score"] +
        0.25 * df["speech_score"] +
        0.25 * df["facial_score"]
    )
    return df

def analyze_correlations(df):
    """
    Calculates the Pearson correlation matrix for the scores.
    """
    score_cols = ["accuracy_score", "speech_score", "facial_score", "overall_score"]
    corr_matrix = df[score_cols].corr()
    print("\n--- Pearson Correlation Matrix ---")
    print(corr_matrix)
    return corr_matrix

def detect_outliers_iqr(df, col="overall_score"):
    """
    Identifies outliers in overall_score using the IQR method.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n--- Outlier Detection on '{col}' (Bounds: {lower_bound:.3f} to {upper_bound:.3f}) ---")
    if outliers.empty:
        print("No outliers detected in the normalized dataset.")
    else:
        print(f"Detected {len(outliers)} outlier(s):")
        print(outliers)
    return outliers

def main():
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.path.abspath(os.path.join(os.getcwd(), "analytics"))
        if not os.path.exists(os.path.join(script_dir, "sample_data")):
            script_dir = os.getcwd()
            
    # Step 1: Generate dataset
    raw_df = generate_multi_session_mock_data()
    
    # Step 2: Normalize scores to 0-1 range
    normalized_df = normalize_scores(raw_df)
    
    # Step 3: Compute Overall Score
    normalized_df = compute_overall_score(normalized_df)
    
    # Step 4: Pearson Correlation
    corr_matrix = analyze_correlations(normalized_df)
    
    # Step 5: Outlier Detection
    detect_outliers_iqr(normalized_df)
    
    # Step 6: Save outputs
    sample_data_dir = os.path.join(script_dir, "sample_data")
    os.makedirs(sample_data_dir, exist_ok=True)
    
    output_path = os.path.join(sample_data_dir, "normalized_scores.csv")
    normalized_df.to_csv(output_path, index=False)
    print(f"\nNormalized score dataset successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
