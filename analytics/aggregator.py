import pandas as pd

# Read mock scores data
df = pd.read_csv("sample_data/mock_scores.csv")

# Calculate overall score
df["overall_score"] = (
    0.5 * df["accuracy_score"] +
    0.25 * df["speech_score"] +
    0.25 * df["facial_score"]
)

# Generate feedback
def generate_feedback(score):
    if score >= 80:
        return "Excellent performance"
    elif score >= 60:
        return "Good performance, needs minor improvement"
    else:
        return "Needs more practice"

df["feedback"] = df["overall_score"].apply(generate_feedback)

# Save final output
df.to_csv("sample_data/final_scores.csv", index=False)

# Show result
print(df.head())