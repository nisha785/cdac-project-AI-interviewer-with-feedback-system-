from pydantic import BaseModel


class QuestionCreate(BaseModel):
    session_id: int
    question_text: str

class QuestionUpdate(BaseModel):
    question_text: str

class QuestionResponse(BaseModel):
    question_id: int
    session_id: int
    question_text: str

    class Config:
        from_attributes = True

