import json
import os
import pandas as pd
import numpy as np


def load_json(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def to_dataframe(records: list[dict], prefix: str, id_key: str = "id") -> pd.DataFrame:
    """
    Convert a list of score dicts into a DataFrame, prefixing non-id columns
    so merged columns from different sources don't collide (e.g. 'accuracy_score'
    vs 'speech_score').
    """
    df = pd.DataFrame(records)
    df = df.rename(columns={
        col: f"{prefix}_{col}" for col in df.columns if col != id_key
    })
    return df


def merge_session_scores(
    accuracy_records: list[dict],
    speech_records: list[dict],
    facial_records: list[dict]
) -> pd.DataFrame:
    """
    Merge accuracy, speech, and facial score records into a single DataFrame,
    joined on question id.
    """
    df_accuracy = to_dataframe(accuracy_records, "accuracy")
    df_speech = to_dataframe(speech_records, "speech")
    df_facial = to_dataframe(facial_records, "facial")

    merged = df_accuracy.merge(df_speech, on="id", how="outer")
    merged = merged.merge(df_facial, on="id", how="outer")
    merged = merged.sort_values("id").reset_index(drop=True)

    return merged


def compute_dimension_stats(df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    """
    Compute mean/min/max/trend for each numeric scoring dimension across
    all questions in the session.

    'trend' is the slope of a simple linear fit (score vs question order) —
    positive means the candidate improved as the interview progressed,
    negative means performance declined.
    """
    stats = {}
    question_order = np.arange(len(df))

    for col in numeric_columns:
        if col not in df.columns:
            continue

        values = df[col].dropna()
        if len(values) == 0:
            continue

        col_stats = {
            "mean": round(float(values.mean()), 3),
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3),
        }

        # Trend: linear regression slope (only if enough points)
        if len(values) >= 2:
            valid_order = question_order[df[col].notna()]
            slope = np.polyfit(valid_order, values, 1)[0]
            col_stats["trend"] = round(float(slope), 4)
        else:
            col_stats["trend"] = 0.0

        stats[col] = col_stats

    return stats


def calibrate_thresholds(
    df: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Flag which questions fall below a given threshold on each dimension.
    Adds a '<column>_flag' boolean column: True if below threshold (needs attention).
    """
    flagged_df = df.copy()
    for col in numeric_columns:
        if col in flagged_df.columns:
            flagged_df[f"{col}_flag"] = flagged_df[col] < threshold
    return flagged_df


def build_session_report(
    accuracy_path: str,
    speech_path: str,
    facial_path: str,
    threshold: float = 0.5
) -> dict:
    accuracy_records = load_json(accuracy_path)
    speech_records = load_json(speech_path)
    facial_records = load_json(facial_path)

    merged_df = merge_session_scores(accuracy_records, speech_records, facial_records)

    numeric_columns = [
        "accuracy_weighted_accuracy_score",
        "speech_confidence_score",
        "speech_clarity",
        "facial_emotion_score",
        "facial_eye_contact_ratio"
    ]

    dimension_stats = compute_dimension_stats(merged_df, numeric_columns)
    flagged_df = calibrate_thresholds(merged_df, numeric_columns, threshold)

    report = {
        "session_summary": dimension_stats,
        "per_question": flagged_df.to_dict(orient="records")
    }

    return report


if __name__ == "__main__":
    # Real accuracy scores from Stage 4 answer scoring
    accuracy_path = "data/sample_resumes/scoring_output.json"

    # MOCK speech and facial scores — placeholders until those modules are built.
    # Replace these paths with real output from the speech/prosody and facial CNN
    # modules once built; the aggregator logic itself won't need to change.
    speech_path = "data/sample_resumes/mock_speech_scores.json"
    facial_path = "data/sample_resumes/mock_facial_scores.json"

    # Generate mock speech/facial data matching however many questions exist
    with open(accuracy_path, "r") as f:
        accuracy_data = json.load(f)
    question_ids = [q["id"] for q in accuracy_data]

    mock_speech = [
        {
            "id": qid,
            "confidence_score": round(np.random.uniform(0.5, 0.95), 3),
            "clarity": round(np.random.uniform(0.5, 0.95), 3),
            "filler_word_freq": round(np.random.uniform(0.0, 0.2), 3),
            "pace_wpm": round(np.random.uniform(110, 160), 1)
        }
        for qid in question_ids
    ]
    mock_facial = [
        {
            "id": qid,
            "emotion_score": round(np.random.uniform(0.4, 0.9), 3),
            "eye_contact_ratio": round(np.random.uniform(0.5, 0.95), 3)
        }
        for qid in question_ids
    ]

    os.makedirs("data/sample_resumes", exist_ok=True)
    with open(speech_path, "w") as f:
        json.dump(mock_speech, f, indent=2)
    with open(facial_path, "w") as f:
        json.dump(mock_facial, f, indent=2)

    report = build_session_report(accuracy_path, speech_path, facial_path)

    print(json.dumps(report["session_summary"], indent=2))

    with open("data/sample_resumes/session_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved full session report to data/sample_resumes/session_report.json")