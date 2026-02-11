# app/system_models/encounter_model/encounter_model.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database.connection import Base
from app.helpers.time import utcnow

class Encounter(Base):
    __tablename__ = "encounters"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    facility_id = Column(Integer, ForeignKey("facilities.id"), nullable=False)
    provider_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    reason_for_visit = Column(String, nullable=True)
    encounter_type = Column(String, default="outpatient")  # outpatient, inpatient, emergency
    started_at = Column(DateTime(timezone=True), default=utcnow)
    closed_at = Column(DateTime(timezone=True), nullable=True)

    patient = relationship("Patient", back_populates="encounters")
    provider = relationship("User")
    diagnoses = relationship("Diagnosis", back_populates="encounter", cascade="all, delete-orphan")
    prescriptions = relationship("Prescription", back_populates="encounter", cascade="all, delete-orphan")
    notes = relationship("ClinicalNote", back_populates="encounter", cascade="all, delete-orphan")
