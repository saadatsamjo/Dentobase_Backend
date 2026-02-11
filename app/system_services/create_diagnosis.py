from sqlalchemy.ext.asyncio import AsyncSession
from app.system_models.diagnosis_model.diagnosis_model import Diagnosis
from app.system_models.diagnosis_model.diagnosis_schemas import DiagnosisCreate

async def create_diagnosis(db: AsyncSession, diagnosis: DiagnosisCreate):
    """Create a new diagnosis."""
    db_diagnosis = Diagnosis(**diagnosis.model_dump())
    db.add(db_diagnosis)
    await db.commit()
    await db.refresh(db_diagnosis)
    return db_diagnosis