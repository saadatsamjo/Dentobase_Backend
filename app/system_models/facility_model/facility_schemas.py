# app/system_models/facility_model/facility_schemas.py
from typing import Optional, List, Literal
from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator

# facilities
class FacilityBase(BaseModel):
    name: str
    address: Optional[str] = None
    phone: Optional[str] = None

class FacilityCreate(FacilityBase):
    pass

class FacilityUpdate(FacilityBase):
    name: Optional[str] = None

class FacilityResponse(FacilityBase):
    id: int
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True
