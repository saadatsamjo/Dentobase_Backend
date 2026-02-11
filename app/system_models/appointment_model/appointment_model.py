# app/system_models/appointment_model/appointment_model.py
from sqlalchemy import Column, Integer, String, Date, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.database.connection import Base
from datetime import datetime

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    appointment_date = Column(Date)
    appointment_time = Column(String)
    appointment_type = Column(String)
    appointment_status = Column(String)
    appointment_notes = Column(Text)
    facility_id = Column(Integer, ForeignKey("facilities.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    patient = relationship("Patient", back_populates="appointments")
    facility = relationship("Facility", back_populates="appointments")