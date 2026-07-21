from app.database.connection import engine
from app.models.base import Base

import app.models.session
import app.models.question
import app.models.answer
import app.models.score

Base.metadata.create_all(bind=engine)

print("Tables created successfully")