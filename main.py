from src.resume_parser import parse_resume
from src.question_generator import generate_questions
import json
import os

def run_pipeline(pdf_path: str, output_path: str = "data/sample_resumes/questions_output.json"):
    print("Parsing resume...")
    resume_data = parse_resume(pdf_path)
    print(f"Extracted skills: {resume_data['skills']}")

    print("\nGenerating questions...")
    questions = generate_questions(resume_data)

    print(json.dumps(questions, indent=2))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nSaved questions to {output_path}")

    return questions

if __name__ == "__main__":
    run_pipeline("data/sample_resumes/test_resume.pdf")