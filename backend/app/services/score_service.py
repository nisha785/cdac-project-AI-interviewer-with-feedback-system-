from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.score import Score
from app.schemas.score_schema import ScoreCreate, ScoreUpdate


def create_score(db: Session, score: ScoreCreate):

    db_score = Score(
        answer_id=score.answer_id,
        accuracy_score=score.accuracy_score,
        speech_score=score.speech_score,
        facial_score=score.facial_score
    )

    db.add(db_score)
    db.commit()
    db.refresh(db_score)

    return db_score


def get_all_scores(db: Session):

    return db.query(Score).all()


def get_score_by_id(db: Session, score_id: int):

    score = (
        db.query(Score)
        .filter(Score.score_id == score_id)
        .first()
    )

    if score is None:
        raise HTTPException(
            status_code=404,
            detail="Score not found"
        )

    return score


def update_score(
    db: Session,
    score_id: int,
    updated_score: ScoreUpdate
):

    score = (
        db.query(Score)
        .filter(Score.score_id == score_id)
        .first()
    )

    if score is None:
        raise HTTPException(
            status_code=404,
            detail="Score not found"
        )

    score.accuracy_score = updated_score.accuracy_score
    score.speech_score = updated_score.speech_score
    score.facial_score = updated_score.facial_score

    db.commit()
    db.refresh(score)

    return score


def delete_score(db: Session, score_id: int):

    score = (
        db.query(Score)
        .filter(Score.score_id == score_id)
        .first()
    )

    if score is None:
        raise HTTPException(
            status_code=404,
            detail="Score not found"
        )

    db.delete(score)
    db.commit()

    return score