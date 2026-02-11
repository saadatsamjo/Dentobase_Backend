# app/system_services/create_facility.py
from sqlalchemy.ext.asyncio import AsyncSession
from app.system_models.facility_model.facility_model import Facility
from app.system_models.facility_model.facility_schemas import FacilityCreate

async def create_facility(db: AsyncSession, facility: FacilityCreate):
    """Create a new facility."""
    db_facility = Facility(**facility.model_dump())
    db.add(db_facility)
    await db.commit()
    await db.refresh(db_facility)
    return db_facility