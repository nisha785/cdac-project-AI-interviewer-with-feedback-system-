from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.database.connection import get_db
from app.schemas.question_schema import QuestionCreate, QuestionResponse,QuestionUpdate
from app.services.question_service import create_question, get_all_questions,get_question_by_id,update_question,delete_question
from app.security.auth import get_current_admin
router = APIRouter()


@router.post("/")
def add_question(
    question: QuestionCreate,
    db: Session = Depends(get_db),
    admin = Depends(get_current_admin)
):
    return create_question(db, question)

@router.get("/", response_model=List[QuestionResponse])
def get_questions(
    db: Session = Depends(get_db)
):
    return get_all_questions(db)

@router.get("/{question_id}", response_model=QuestionResponse)
def get_question(
    question_id: int,
    db: Session = Depends(get_db)
):
    return get_question_by_id(db, question_id)

@router.put("/{question_id}", response_model=QuestionResponse)
def update_existing_question(
    question_id: int,
    question: QuestionUpdate,
    db: Session = Depends(get_db),
    admin = Depends(get_current_admin)
):
    return update_question(db, question_id, question)


@router.delete("/{question_id}")
def remove_question(
    question_id: int,
    db: Session = Depends(get_db),
    admin = Depends(get_current_admin)
):
    return delete_question(db, question_id)