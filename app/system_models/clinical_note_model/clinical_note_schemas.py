# app/system_models/clinical_note_model/clinical_note_schemas.py
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class ClinicalNoteBase(BaseModel):
    encounter_id: int
    author_id: int
    note_type: str = "SOAP"
    content: str

class ClinicalNoteCreate(ClinicalNoteBase):
    pass

class ClinicalNoteUpdate(BaseModel):
    note_type: Optional[str] = None
    content: Optional[str] = None

class ClinicalNoteResponse(ClinicalNoteBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True