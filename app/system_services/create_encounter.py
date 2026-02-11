from sqlalchemy.ext.asyncio import AsyncSession
from app.system_models.encounter_model.encounter_model import Encounter
from app.system_models.encounter_model.encounter_schemas import EncounterCreate

async def create_encounter(db: AsyncSession, encounter: EncounterCreate):
    """Create a new encounter."""
    db_encounter = Encounter(**encounter.model_dump())
    db.add(db_encounter)
    await db.commit()
    await db.refresh(db_encounter)
    return db_encounter