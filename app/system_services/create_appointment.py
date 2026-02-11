from sqlalchemy.ext.asyncio import AsyncSession
from app.system_models.appointment_model.appointment_model import Appointment
from app.system_models.appointment_model.appointment_schemas import AppointmentCreate

async def create_appointment(db: AsyncSession, appointment: AppointmentCreate):
    """Create a new appointment."""
    db_appointment = Appointment(**appointment.model_dump())
    db.add(db_appointment)
    await db.commit()
    await db.refresh(db_appointment)
    return db_appointment