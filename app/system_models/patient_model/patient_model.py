# app/system_models/patient_model/patient_model.py
from sqlalchemy import Column, Integer, String, Date, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from app.database.connection import Base
from app.helpers.time import utcnow

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)

    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    dob = Column(Date, nullable=True)
    gender = Column(String, nullable=True)
    contact_info = Column(String, nullable=True)

    facility_id = Column(Integer, ForeignKey("facilities.id"), nullable=False)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        CheckConstraint("gender IN ('male', 'female', 'other')", name="check_gender_values"),
    )

    facility = relationship("Facility", back_populates="patients")
    encounters = relationship("Encounter", back_populates="patient", cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="patient")

    def __repr__(self):
        return f"<Patient {self.id}: {self.first_name} {self.last_name}>"
