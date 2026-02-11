from sqlalchemy.ext.asyncio import AsyncSession
from app.system_models.prescription_model.prescription_model import Prescription
from app.system_models.prescription_model.prescription_schemas import PrescriptionCreate

async def create_prescription(db: AsyncSession, prescription: PrescriptionCreate):
    """Create a new prescription."""
    db_prescription = Prescription(**prescription.model_dump())
    db.add(db_prescription)
    await db.commit()
    await db.refresh(db_prescription)
    return db_prescription