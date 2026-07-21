from sqlalchemy.orm import Session

from app.models.question import Question
from app.schemas.question_schema import QuestionCreate,QuestionUpdate


def create_question(db: Session, question: QuestionCreate):

    db_question = Question(
        session_id=question.session_id,
        question_text=question.question_text
    )

    db.add(db_question)

    db.commit()

    db.refresh(db_question)

    return db_question

def get_all_questions(db):
    return db.query(Question).all()


def get_question_by_id(db: Session, question_id: int):
    return (
        db.query(Question)
        .filter(Question.question_id == question_id)
        .first()
    )

def update_question(
    db: Session,
    question_id: int,
    updated_question: QuestionUpdate
):
    question = (
        db.query(Question)
        .filter(Question.question_id == question_id)
        .first()
    )

    if question:
        question.question_text = updated_question.question_text

        db.commit()
        db.refresh(question)

    return question

def delete_question(db: Session, question_id: int):

    question = (
        db.query(Question)
        .filter(Question.question_id == question_id)
        .first()
    )

    if question:
        db.delete(question)
        db.commit()

    return question