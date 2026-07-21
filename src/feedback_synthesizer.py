import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_feedback_prompt(session_report: dict) -> str:
    """
    Build a structured prompt from the aggregated session report (output of
    session_aggregator.py), asking the LLM to synthesize feedback across
    communication, answer quality, body language, and an action plan.
    """
    summary = session_report["session_summary"]
    per_question = session_report["per_question"]

    flagged_questions = []
    for q in per_question:
        flags = [k for k, v in q.items() if k.endswith("_flag") and v is True]
        if flags:
            flagged_questions.append({
                "id": q.get("id"),
                "weak_areas": [f.replace("_flag", "") for f in flags]
            })

    return f"""You are an interview coach analyzing a candidate's mock interview performance.
Below is aggregated scoring data across three dimensions: answer accuracy, speech/prosody,
and facial/body language, collected across all questions in the session.

SESSION-LEVEL STATS (mean/min/max/trend per dimension):
{json.dumps(summary, indent=2)}

QUESTIONS THAT FELL BELOW THE PERFORMANCE THRESHOLD:
{json.dumps(flagged_questions, indent=2)}

Based on this data, write a structured feedback report. Return ONLY valid JSON
in this exact format, no preamble, no markdown fences:

{{
  "communication": {{
    "summary": "2-3 sentence assessment of clarity and confidence based on speech stats",
    "score_out_of_10": 0
  }},
  "answer_quality": {{
    "summary": "2-3 sentence assessment of accuracy and relevance based on accuracy stats",
    "score_out_of_10": 0
  }},
  "body_language": {{
    "summary": "2-3 sentence assessment of eye contact and emotional expression based on facial stats",
    "score_out_of_10": 0
  }},
  "overall_trend": "1-2 sentences describing whether performance improved or declined across the session, referencing the trend values",
  "action_plan": [
    "specific, actionable improvement point 1",
    "specific, actionable improvement point 2",
    "specific, actionable improvement point 3"
  ]
}}

Be specific and reference actual numbers from the data where relevant. Avoid generic
advice — ground every point in the stats provided. Do not be generic or vague."""


def generate_feedback(session_report: dict) -> dict:
    prompt = build_feedback_prompt(session_report)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.choices[0].message.content.strip()
    response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print("Failed to parse LLM output as JSON. Raw output:")
        print(response_text)
        raise


if __name__ == "__main__":
    # This module only reads the already-built session report — it does not
    # generate or regenerate any scores itself.
    session_report_path = "data/sample_resumes/session_report.json"

    with open(session_report_path, "r") as f:
        session_report = json.load(f)

    feedback = generate_feedback(session_report)

    print(json.dumps(feedback, indent=2))

    with open("data/sample_resumes/feedback_report.json", "w") as f:
        json.dump(feedback, f, indent=2)

    print("\nSaved feedback report to data/sample_resumes/feedback_report.json")