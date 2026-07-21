from sqlalchemy.orm import Session
from app.models.session import Session as SessionModel


def create_session(db: Session, candidate_name: str):
    new_session = SessionModel(
        candidate_name=candidate_name,
        status="started"
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return new_session


