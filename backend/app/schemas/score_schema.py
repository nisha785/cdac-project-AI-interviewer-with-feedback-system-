from pydantic import BaseModel


class ScoreCreate(BaseModel):
    answer_id: int
    accuracy_score: float
    speech_score: float
    facial_score: float


class ScoreUpdate(BaseModel):
    accuracy_score: float
    speech_score: float
    facial_score: float


class ScoreResponse(BaseModel):
    score_id: int
    answer_id: int
    accuracy_score: float
    speech_score: float
    facial_score: float

    class Config:
        from_attributes = True