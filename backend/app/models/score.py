from sqlalchemy import Column, Integer, Float, ForeignKey
from .base import Base

class Score(Base):
    __tablename__ = "scores"

    score_id = Column(Integer, primary_key=True, index=True)

    answer_id = Column(
        Integer,
        ForeignKey("answers.answer_id")
    )

    accuracy_score = Column(Float)

    speech_score = Column(Float)

    facial_score = Column(Float)