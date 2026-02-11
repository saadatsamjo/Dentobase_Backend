# app/system_models/facility_model/facility_model.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from app.database.connection import Base
from app.helpers.time import utcnow

class Facility(Base):
    __tablename__ = "facilities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    address = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    users = relationship("User", back_populates="facility")
    patients = relationship("Patient", back_populates="facility")
    appointments = relationship("Appointment", back_populates="facility")

    def __repr__(self):
        return f"<Facility(id={self.id}, name='{self.name}')>"
