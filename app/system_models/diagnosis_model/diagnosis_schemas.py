# app/system_models/diagnosis_model/diagnosis_schemas.py
from typing import Optional
from pydantic import BaseModel

class DiagnosisBase(BaseModel):
    encounter_id: int
    code: Optional[str] = None
    description: str
    diagnosis_type: str = "primary"

class DiagnosisCreate(DiagnosisBase):
    pass

class DiagnosisUpdate(BaseModel):
    code: Optional[str] = None
    description: Optional[str] = None
    diagnosis_type: Optional[str] = None

class DiagnosisResponse(DiagnosisBase):
    id: int

    class Config:
        from_attributes = True
