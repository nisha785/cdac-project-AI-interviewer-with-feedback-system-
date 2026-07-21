from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session


from app.database.connection import get_db
from app.schemas.user_schema import (
    UserCreate,
    UserLogin,
    UserResponse,
    LoginResponse
)
from app.services.user_service import register_user, login_user


router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(
    user: UserCreate,
    db: Session = Depends(get_db)
):
    return register_user(db, user)

@router.post("/login", response_model=LoginResponse)
def login(
    user: UserLogin,
    db: Session = Depends(get_db)
):
    return login_user(db, user)