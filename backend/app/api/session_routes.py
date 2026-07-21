from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.schemas.session_schema import SessionCreate
from app.services.session_service import create_session
from app.security.auth import get_current_admin

router = APIRouter()


@router.get("/")
def home():
    return {"message": "Session API"}


@router.post("/")
def create_new_session(
    session: SessionCreate,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin)
):
    return create_session(db, session.candidate_name)