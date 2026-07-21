from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.security.auth import get_current_user

from app.database.connection import get_db
from app.schemas.score_schema import (
    ScoreCreate,
    ScoreUpdate,
    ScoreResponse
)
from app.services.score_service import (
    create_score,
    get_all_scores,
    get_score_by_id,
    update_score,
    delete_score
)

router = APIRouter()


@router.post("/", response_model=ScoreResponse)
def add_score(
    score: ScoreCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return create_score(db, score)

@router.get("/", response_model=List[ScoreResponse])
def get_scores(
    db: Session = Depends(get_db)
):
    return get_all_scores(db)


@router.get("/{score_id}", response_model=ScoreResponse)
def get_score(
    score_id: int,
    db: Session = Depends(get_db)
):
    return get_score_by_id(db, score_id)


@router.put("/{score_id}", response_model=ScoreResponse)
def edit_score(
    score_id: int,
    updated_score: ScoreUpdate,
    db: Session = Depends(get_db)
):
    return update_score(db, score_id, updated_score)


@router.delete("/{score_id}", response_model=ScoreResponse)
def remove_score(
    score_id: int,
    db: Session = Depends(get_db)
):
    return delete_score(db, score_id)