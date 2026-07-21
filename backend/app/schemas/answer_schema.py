from pydantic import BaseModel


class AnswerCreate(BaseModel):
    question_id: int
    transcript: str


class AnswerUpdate(BaseModel):
    transcript: str


class AnswerResponse(BaseModel):
    answer_id: int
    question_id: int
    transcript: str

    class Config:
        from_attributes = True