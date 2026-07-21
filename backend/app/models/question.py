from sqlalchemy import Column, Integer, Text, ForeignKey
from .base import Base

class Question(Base):
    __tablename__ = "questions"

    question_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.session_id"))
    question_text = Column(Text)