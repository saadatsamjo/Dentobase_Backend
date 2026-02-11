# app/system_models/encounter_model/encounter_schemas.py
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class EncounterBase(BaseModel):
    patient_id: int
    facility_id: int
    provider_id: int
    reason_for_visit: Optional[str]
    encounter_type: str = "outpatient"

class EncounterCreate(EncounterBase):
    pass

class EncounterResponse(EncounterBase):
    id: int
    started_at: datetime
    closed_at: Optional[datetime]

    class Config:
        from_attributes = True
