from sqlalchemy import JSON, Column, Float, String, Text

from app.db.base_class import Base


class Question(Base):
    __tablename__ = "questions"  # type: ignore
    id = Column(String, primary_key=True)
    question_text = Column(Text, nullable=False)
    category = Column(String)
    provider = Column(String)
    cost = Column(Float)
    tokens = Column(JSON)
    question_metadata = Column(JSON)
