from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.security.auth import get_current_user

from app.database.connection import get_db
from app.schemas.answer_schema import (
    AnswerCreate,
    AnswerUpdate,
    AnswerResponse
)
from app.services.answer_service import (
    create_answer,
    get_all_answers,
    get_answer_by_id,
    update_answer,
    delete_answer
)

router = APIRouter()

@router.post("/", response_model=AnswerResponse)
def add_answer(
    answer: AnswerCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return create_answer(db, answer)

@router.get("/", response_model=List[AnswerResponse])
def get_answers(
    db: Session = Depends(get_db)
):
    return get_all_answers(db)

@router.get("/{answer_id}", response_model=AnswerResponse)
def get_answer(
    answer_id: int,
    db: Session = Depends(get_db)
):
    return get_answer_by_id(db, answer_id)

@router.put("/{answer_id}", response_model=AnswerResponse)
def edit_answer(
    answer_id: int,
    updated_answer: AnswerUpdate,
    db: Session = Depends(get_db)
):
    return update_answer(db, answer_id, updated_answer)

@router.delete("/{answer_id}", response_model=AnswerResponse)
def remove_answer(
    answer_id: int,
    db: Session = Depends(get_db)
):
    return delete_answer(db, answer_id)