# app/system_models/diagnosis_model/diagnosis_model.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.database.connection import Base

class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True)
    encounter_id = Column(Integer, ForeignKey("encounters.id"), nullable=False)

    code = Column(String, nullable=True)  # ICD-10 / SNOMED
    description = Column(String, nullable=False)
    diagnosis_type = Column(String, default="primary")

    encounter = relationship("Encounter", back_populates="diagnoses")
