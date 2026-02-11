# app/system_services/create_patient.py
from sqlalchemy.ext.asyncio import AsyncSession
from app.system_models.patient_model.patient_model import Patient
from app.system_models.patient_model.patient_schemas import PatientCreate

async def create_patient(db: AsyncSession, patient: PatientCreate):
    """Create a new patient."""
    db_patient = Patient(**patient.model_dump())
    db.add(db_patient)
    await db.commit()
    await db.refresh(db_patient)
    return db_patient
