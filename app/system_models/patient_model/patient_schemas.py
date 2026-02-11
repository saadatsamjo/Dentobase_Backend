# app/system_models/patient_model/patient_schemas.py
from typing import Optional, Literal
from datetime import date, datetime
from pydantic import BaseModel, field_validator

GENDER = Literal["male", "female", "other"]

class PatientBase(BaseModel):
    first_name: str
    last_name: str
    dob: Optional[date]
    gender: Optional[GENDER]
    contact_info: Optional[str]
    facility_id: int

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    dob: Optional[date]
    gender: Optional[GENDER]
    contact_info: Optional[str]

class PatientResponse(PatientBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
