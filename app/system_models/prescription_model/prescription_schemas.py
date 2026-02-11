# app/system_models/prescription_model/prescription_schemas.py
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class PrescriptionBase(BaseModel):
    encounter_id: int
    drug_name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None

class PrescriptionCreate(PrescriptionBase):
    pass

class PrescriptionUpdate(BaseModel):
    drug_name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None

class PrescriptionResponse(PrescriptionBase):
    id: int
    prescribed_at: datetime

    class Config:
        from_attributes = True
