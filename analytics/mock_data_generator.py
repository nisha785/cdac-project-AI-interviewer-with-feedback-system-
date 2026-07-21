import pandas as pd
import random

data = []

for session_id in range(1, 6):
    for question_id in range(1, 6):
        data.append({
            "session_id": session_id,
            "question_id": question_id,
            "accuracy_score": random.randint(40, 100),
            "speech_score": random.randint(40, 100),
            "facial_score": random.randint(40, 100)
        })

df = pd.DataFrame(data)

df.to_csv("sample_data/mock_scores.csv", index=False)

print(df.head())