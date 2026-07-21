from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.models.answer import Answer
from app.schemas.answer_schema import AnswerCreate,AnswerUpdate

def create_answer(db: Session, answer: AnswerCreate):

    db_answer = Answer(
        question_id=answer.question_id,
        transcript=answer.transcript
    )

    db.add(db_answer)

    db.commit()

    db.refresh(db_answer)

    return db_answer

def get_all_answers(db: Session):
    return db.query(Answer).all()

def get_answer_by_id(db: Session, answer_id: int):

    answer = (
        db.query(Answer)
        .filter(Answer.answer_id == answer_id)
        .first()
    )

    if answer is None:
        raise HTTPException(
            status_code=404,
            detail="Answer not found"
        )

    return answer

def update_answer(
    db: Session,
    answer_id: int,
    updated_answer: AnswerUpdate
):
    answer = (
        db.query(Answer)
        .filter(Answer.answer_id == answer_id)
        .first()
    )

    if answer is None:
        raise HTTPException(
            status_code=404,
            detail="Answer not found"
        )

    answer.transcript = updated_answer.transcript

    db.commit()
    db.refresh(answer)

    return answer

def delete_answer(db: Session, answer_id: int):

    answer = (
        db.query(Answer)
        .filter(Answer.answer_id == answer_id)
        .first()
    )

    if answer is None:
        raise HTTPException(
            status_code=404,
            detail="Answer not found"
        )

    db.delete(answer)
    db.commit()

    return answer