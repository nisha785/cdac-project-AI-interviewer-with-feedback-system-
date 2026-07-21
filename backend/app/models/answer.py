from sqlalchemy import Column, Integer, Text, ForeignKey
from .base import Base

class Answer(Base):
    __tablename__ = "answers"

    answer_id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.question_id"))
    transcript = Column(Text)