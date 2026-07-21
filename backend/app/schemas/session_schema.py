from pydantic import BaseModel

class SessionCreate(BaseModel):
    candidate_name: str