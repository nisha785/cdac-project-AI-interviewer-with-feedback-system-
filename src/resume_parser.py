import pdfplumber
import spacy
import json
import re

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a resume PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_email(text: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None

def extract_phone(text: str) -> str | None:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\d{10}", text)
    return match.group(0) if match else None

def extract_skills(text: str, skill_keywords: list[str]) -> list[str]:
    """Simple keyword-matching skill extraction. Expand skill_keywords as needed."""
    found = []
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill.lower() in text_lower:
            found.append(skill)
    return found

def parse_resume(pdf_path: str) -> dict:
    """Main entry point: PDF -> structured resume JSON."""
    raw_text = extract_text_from_pdf(pdf_path)
    doc = nlp(raw_text)

    # Common skill keywords — extend this list based on your domain
    skill_keywords = [
        "Python", "Java", "JavaScript", "SQL", "Machine Learning", "Deep Learning",
        "TensorFlow", "PyTorch", "React", "Node.js", "AWS", "Docker", "Kubernetes",
        "FastAPI", "Flask", "Django", "Git", "NLP", "Computer Vision", "Pandas",
        "NumPy", "Scikit-learn", "XGBoost", "MongoDB", "PostgreSQL", "REST API"
    ]

    # Named entities (org names, dates) via spaCy — useful for experience/education parsing
    orgs = list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))
    dates = list(set(ent.text for ent in doc.ents if ent.label_ == "DATE"))

    resume_json = {
        "email": extract_email(raw_text),
        "phone": extract_phone(raw_text),
        "skills": extract_skills(raw_text, skill_keywords),
        "organizations": orgs,
        "dates_mentioned": dates,
        "raw_text": raw_text
    }
    return resume_json

if __name__ == "__main__":
    result = parse_resume("data/sample_resumes/test_resume.pdf")
    print(json.dumps(result, indent=2))