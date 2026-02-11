from sqlalchemy.ext.asyncio import AsyncSession
from app.system_models.clinical_note_model.clinical_note_model import ClinicalNote
from app.system_models.clinical_note_model.clinical_note_schemas import ClinicalNoteCreate

async def create_clinical_note(db: AsyncSession, clinical_note: ClinicalNoteCreate):
    """Create a new clinical note."""
    db_clinical_note = ClinicalNote(**clinical_note.model_dump())
    db.add(db_clinical_note)
    await db.commit()
    await db.refresh(db_clinical_note)
    return db_clinical_note