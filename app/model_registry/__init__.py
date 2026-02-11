# app/model_registry/__init__.py


# Register all models here

# User models
from app.users.user_models.user_model import User
from app.users.auth_token_model.token_model import Token
from app.users.password_reset_token.password_reset_token_model import PasswordResetToken

# System models
from app.system_models.patient_model.patient_model import Patient 
from app.system_models.facility_model.facility_model import Facility
from app.system_models.appointment_model.appointment_model import Appointment
from app.system_models.encounter_model.encounter_model import Encounter
from app.system_models.diagnosis_model.diagnosis_model import Diagnosis
from app.system_models.prescription_model.prescription_model import Prescription
from app.system_models.clinical_note_model.clinical_note_model import ClinicalNote