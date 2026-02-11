# app/system_models/clinical_note_model/clinical_note_model.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database.connection import Base
from app.helpers.time import utcnow

class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id = Column(Integer, primary_key=True)
    encounter_id = Column(Integer, ForeignKey("encounters.id"), nullable=False)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    note_type = Column(String, default="SOAP")
    content = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    encounter = relationship("Encounter", back_populates="notes")
    author = relationship("User")
