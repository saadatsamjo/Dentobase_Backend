# app/users/user_models/user_model.py
from typing import Optional
from pydantic import BaseModel, Field
from config.appconfig import settings
from sqlalchemy import Column, Integer, String, Boolean, DateTime, CheckConstraint, ForeignKey
from sqlalchemy.orm import relationship
from app.database.connection import Base
from app.helpers.time import utcnow




class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    verification_code = Column(String, nullable=True)
    role = Column(String, default="doctor", nullable=False)
    
    facility_id = Column(Integer, ForeignKey("facilities.id"), nullable=True)
    facility = relationship("Facility", back_populates="users")
    tokens = relationship("Token", back_populates="user")
    reset_tokens = relationship("PasswordResetToken", back_populates="user")

    # Optional: Add more fields as needed
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    


    # Add check constraints for validation at database level
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'doctor')", name='check_role_values'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, first_name='{self.first_name}', last_name='{self.last_name}', email='{self.email}')>"