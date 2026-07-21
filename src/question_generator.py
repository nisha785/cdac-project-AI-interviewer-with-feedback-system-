import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(resume_json: dict) -> str:
    return f"""You are an interview question generator. Based on the resume data below,
generate exactly 10 interview questions ordered from easy to hard, each with an ideal answer.

Resume data:
- Skills: {resume_json.get('skills')}
- Organizations mentioned: {resume_json.get('organizations')}
- Resume text excerpt: {resume_json.get('raw_text', '')[:2000]}

Return ONLY valid JSON in this exact format, no preamble, no markdown fences:
{{
  "questions": [
    {{
      "id": 1,
      "difficulty": "easy",
      "question": "...",
      "ideal_answer": "..."
    }}
  ]
}}
"""

def generate_questions(resume_json: dict) -> dict:
    prompt = build_prompt(resume_json)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.choices[0].message.content.strip()

    # Defensive cleanup in case the model wraps output in code fences
    response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print("Failed to parse LLM output as JSON. Raw output:")
        print(response_text)
        raise

if __name__ == "__main__":
    from resume_parser import parse_resume

    resume_data = parse_resume("data/sample_resumes/test_resume.pdf")
    questions = generate_questions(resume_data)
    print(json.dumps(questions, indent=2))

    with open("data/sample_resumes/questions_output.json", "w") as f:
        json.dump(questions, f, indent=2)