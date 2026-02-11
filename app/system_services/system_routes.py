# app/system_services/system_routes.py
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.system_models.facility_model.facility_schemas import FacilityCreate, FacilityResponse
from app.system_models.patient_model.patient_schemas import PatientCreate, PatientResponse
from app.system_models.appointment_model.appointment_schemas import AppointmentCreate, AppointmentResponse
from app.system_models.clinical_note_model.clinical_note_schemas import ClinicalNoteCreate, ClinicalNoteResponse
from app.system_models.diagnosis_model.diagnosis_schemas import DiagnosisCreate, DiagnosisResponse
from app.system_models.encounter_model.encounter_schemas import EncounterCreate, EncounterResponse
from app.system_models.prescription_model.prescription_schemas import PrescriptionCreate, PrescriptionResponse
from app.system_services.create_facility import create_facility
from app.system_services.create_patient import create_patient
from app.system_services.create_appointment import create_appointment
from app.system_services.create_clinical_note import create_clinical_note
from app.system_services.create_diagnosis import create_diagnosis
from app.system_services.create_encounter import create_encounter
from app.system_services.create_prescription import create_prescription

router = APIRouter()

@router.post("/create_facility", response_model=FacilityResponse)
async def create_facility_endpoint(facility: FacilityCreate, db: AsyncSession = Depends(get_db)):
    """Create a new facility."""
    try:
        db_facility = await create_facility(db, facility)
        return db_facility
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_patient", response_model=PatientResponse)
async def create_patient_endpoint(patient: PatientCreate, db: AsyncSession = Depends(get_db)):
    """Create a new patient."""
    try:
        db_patient = await create_patient(db, patient)
        return db_patient
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_appointment", response_model=AppointmentResponse)
async def create_appointment_endpoint(appointment: AppointmentCreate, db: AsyncSession = Depends(get_db)):
    """Create a new appointment."""
    try:
        db_appointment = await create_appointment(db, appointment)
        return db_appointment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_clinical_note", response_model=ClinicalNoteResponse)
async def create_clinical_note_endpoint(clinical_note: ClinicalNoteCreate, db: AsyncSession = Depends(get_db)):
    """Create a new clinical note."""
    try:
        db_clinical_note = await create_clinical_note(db, clinical_note)
        return db_clinical_note
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_diagnosis", response_model=DiagnosisResponse)
async def create_diagnosis_endpoint(diagnosis: DiagnosisCreate, db: AsyncSession = Depends(get_db)):
    """Create a new diagnosis."""
    try:
        db_diagnosis = await create_diagnosis(db, diagnosis)
        return db_diagnosis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_encounter", response_model=EncounterResponse)
async def create_encounter_endpoint(encounter: EncounterCreate, db: AsyncSession = Depends(get_db)):
    """Create a new encounter."""
    try:
        db_encounter = await create_encounter(db, encounter)
        return db_encounter
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_prescription", response_model=PrescriptionResponse)
async def create_prescription_endpoint(prescription: PrescriptionCreate, db: AsyncSession = Depends(get_db)):
    """Create a new prescription."""
    try:
        db_prescription = await create_prescription(db, prescription)
        return db_prescription
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))