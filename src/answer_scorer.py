import spacy
from sentence_transformers import SentenceTransformer, util

# Load models once at module level (avoid reloading per call — expensive)
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(candidate_answer: str, ideal_answer: str) -> float:
    """
    Compute cosine similarity between candidate and ideal answer embeddings.
    Returns a float in [0, 1] (SBERT cosine sim is typically 0-1 for similar text).
    """
    embeddings = sbert_model.encode([candidate_answer, ideal_answer], convert_to_tensor=True)
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Clamp to [0, 1] in case of minor negative float noise
    return max(0.0, min(1.0, cosine_sim))


def extract_keywords(text: str) -> set[str]:
    """
    Extract key terms from text using spaCy: nouns, proper nouns, and technical
    terms (skipping stopwords/punctuation). Lowercased for matching.
    """
    doc = nlp(text)
    keywords = set()
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and not token.is_punct:
            keywords.add(token.lemma_.lower())
    return keywords


def keyword_coverage(candidate_answer: str, ideal_answer: str) -> float:
    """
    Fraction of the ideal answer's key terms that also appear in the candidate answer.
    Returns 0 if the ideal answer has no extractable keywords (avoids divide-by-zero).
    """
    ideal_keywords = extract_keywords(ideal_answer)
    if not ideal_keywords:
        return 0.0

    candidate_keywords = extract_keywords(candidate_answer)
    matched = ideal_keywords.intersection(candidate_keywords)
    return len(matched) / len(ideal_keywords)


def score_answer(
    question: str,
    candidate_answer: str,
    ideal_answer: str,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> dict:
    """
    Compute a weighted accuracy score (0-1) for a single answer.

    semantic_weight + keyword_weight should sum to 1.0.
    Default 70/30 split favors meaning over exact terminology, since candidates
    often express correct ideas with different words.
    """
    sem_score = semantic_similarity(candidate_answer, ideal_answer)
    kw_score = keyword_coverage(candidate_answer, ideal_answer)

    weighted_score = (semantic_weight * sem_score) + (keyword_weight * kw_score)

    return {
        "question": question,
        "candidate_answer": candidate_answer,
        "semantic_similarity": round(sem_score, 3),
        "keyword_coverage": round(kw_score, 3),
        "weighted_accuracy_score": round(weighted_score, 3)
    }


def score_all_answers(questions_data: dict, candidate_answers: dict) -> list[dict]:
    """
    Score multiple answers at once.

    questions_data: the JSON from question_generator.py, e.g.
        {"questions": [{"id": 1, "question": "...", "ideal_answer": "..."}, ...]}
    candidate_answers: dict mapping question id -> candidate's answer text, e.g.
        {1: "my answer to question 1", 2: "my answer to question 2"}
    """
    results = []
    for q in questions_data["questions"]:
        q_id = q["id"]
        candidate_answer = candidate_answers.get(q_id, "")
        if not candidate_answer:
            continue  # skip unanswered questions

        result = score_answer(
            question=q["question"],
            candidate_answer=candidate_answer,
            ideal_answer=q["ideal_answer"]
        )
        result["id"] = q_id
        result["difficulty"] = q.get("difficulty", "unknown")
        results.append(result)

    return results

if __name__ == "__main__":
    import json

    # Load questions generated in Stage 2
    with open("data/sample_resumes/questions_output.json", "r") as f:
        questions_data = json.load(f)

    # No hardcoded answers — use each question's own ideal_answer as the
    # candidate answer. This is a self-test: since candidate == ideal,
    # every score should land close to 1.0, confirming the scorer works
    # correctly before real transcribed answers (from Whisper) are wired in.
    candidate_answers = {
        q["id"]: q["ideal_answer"] for q in questions_data["questions"]
    }

    results = score_all_answers(questions_data, candidate_answers)

    print(json.dumps(results, indent=2))

    with open("data/sample_resumes/scoring_output.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nScored {len(results)} answers. Saved to data/sample_resumes/scoring_output.json")