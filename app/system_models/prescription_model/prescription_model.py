# app/system_models/prescription_model/prescription_model.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database.connection import Base
from app.helpers.time import utcnow

class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(Integer, primary_key=True)
    encounter_id = Column(Integer, ForeignKey("encounters.id"), nullable=False)

    drug_name = Column(String, nullable=False)
    dosage = Column(String)
    frequency = Column(String)
    duration = Column(String)
    route = Column(String)

    prescribed_at = Column(DateTime(timezone=True), default=utcnow)

    encounter = relationship("Encounter", back_populates="prescriptions")
