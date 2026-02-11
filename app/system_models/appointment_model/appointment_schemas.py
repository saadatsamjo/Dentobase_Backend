# app/system_models/appointment_model/appointment_schemas.py
from typing import Optional, List, Literal
from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator

class AppointmentBase(BaseModel):
    patient_id: int
    appointment_date: date
    appointment_time: str
    appointment_type: str
    appointment_status: str
    appointment_notes: str
    facility_id: Optional[int] = None

class AppointmentCreate(AppointmentBase):
    pass

class AppointmentUpdate(AppointmentBase):
    pass

class AppointmentResponse(AppointmentBase):
    id: int
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True  

