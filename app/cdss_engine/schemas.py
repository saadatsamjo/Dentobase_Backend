# app/cdss_engine/schemas.py
"""
CDSS Request/Response Schemas - Updated with NoImageProvided handling
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Dict, Any
from datetime import datetime

# ============================================================================
# PATIENT HISTORY
# ============================================================================
class PatientHistory(BaseModel):
    """Patient history from database + current complaint."""
    patient_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: Optional[str] = None
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    previous_dental_work: Optional[str] = None
    symptoms_duration: Optional[str] = None
    pain_level: Optional[int] = Field(None, ge=0, le=10)

# ============================================================================
# VISION ANALYSIS
# ============================================================================
class ImageObservation(BaseModel):
    """Structured radiograph analysis from vision model."""
    structured_findings: Optional[Dict[str, Any]] = Field(
        None, 
        description="Highly structured pathology findings (caries, bone loss, etc.)"
    )
    raw_description: str = Field(..., description="Full clinical analysis from vision model")
    pathology_summary: str = Field(..., description="Focused pathology findings")
    focused_tooth: Optional[str] = Field(None, description="Tooth number that was the focus of analysis")
    image_quality_score: float = Field(0.5, description="0-1 score for image quality")
    diagnostic_confidence: float = Field(0.5, description="0-1 score for diagnostic confidence")
    overall_confidence: Literal["high", "medium", "low"] = Field(
        "medium", 
        description="Categorical confidence level",
        alias="confidence"
    )
    model_used: str = Field(..., description="Which vision model was used")
    
    class Config:
        populate_by_name = True

class NoImageProvided(BaseModel):
    """Indicator that no image was provided in the request."""
    message: str = Field(default="No radiograph image was provided for this consultation")
    image_required: bool = Field(default=False, description="Whether image is required for this case")

# ============================================================================
# KNOWLEDGE RETRIEVAL
# ============================================================================
class RetrievedKnowledge(BaseModel):
    """Retrieved clinical guidelines chunk."""
    content: str
    pages: List[int]
    relevance_score: Optional[float] = None
    source: str = "Clinical Guidelines"

# ============================================================================
# CLINICAL RECOMMENDATION
# ============================================================================
class ClinicalRecommendation(BaseModel):
    """Final structured clinical recommendation."""
    diagnosis: str = Field(..., description="Primary clinical diagnosis")
    differential_diagnoses: List[str] = Field(
        default_factory=list,
        description="Alternative diagnoses to consider"
    )
    recommended_management: str = Field(..., description="Treatment plan and next steps")
    reference_pages: List[int] = Field(
        default_factory=list,
        description="Page numbers from guidelines used (integers only, no 'Page X' format)"
    )
    confidence_level: Literal["high", "medium", "low"] = "medium"
    llm_provider: str = Field(default="ollama", description="Which LLM generated this")

# ============================================================================
# CDSS REQUEST
# ============================================================================
class CDSSRequest(BaseModel):
    """Request for CDSS recommendation."""
    patient_id: int = Field(..., description="Patient database ID")
    chief_complaint: str = Field(..., description="Main clinical complaint/question")
    user_id: Optional[int] = Field(1, description="Requesting doctor/admin ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": 123,
                "chief_complaint": "severe tooth pain, sensitivity to cold",
                "user_id": 1
            }
        }

# ============================================================================
# CDSS RESPONSE (Improved Structure)
# ============================================================================
class CDSSResponse(BaseModel):
    """Complete CDSS response with improved organization."""
    
    # === PRIMARY RESULT ===
    recommendation: ClinicalRecommendation
    
    # === SUPPORTING DATA ===
    image_observations: Union[ImageObservation, NoImageProvided] = Field(
        ...,
        description="Radiograph analysis if image was provided, or NoImageProvided message"
    )
    
    knowledge_sources: List[RetrievedKnowledge] = Field(
        default_factory=list,
        description="Retrieved guideline chunks"
    )
    
    # === REASONING TRACE ===
    reasoning_chain: str = Field(
        ...,
        description="Step-by-step reasoning process"
    )
    
    # === METADATA ===
    processing_metadata: dict = Field(
        default_factory=dict,
        description="Performance metrics and model information"
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "recommendation": {
                    "diagnosis": "Irreversible pulpitis, tooth #30",
                    "differential_diagnoses": [
                        "Acute apical periodontitis",
                        "Symptomatic irreversible pulpitis"
                    ],
                    "recommended_management": "Root canal therapy indicated for tooth #30. Consider referral to endodontist given periapical radiolucency.",
                    "reference_pages": [45, 46, 47],
                    "confidence_level": "high",
                    "llm_provider": "gpt-4"
                },
                "image_observations": {
                    "raw_description": "Periapical radiograph shows tooth #30 with large occlusal caries extending into pulp...",
                    "pathology_summary": "Deep caries, periapical radiolucency 4mm",
                    "confidence": "high",
                    "model_used": "gpt-4-vision"
                },
                "processing_metadata": {
                    "total_time_seconds": 8.45,
                    "vision_provider": "gpt4v",
                    "llm_provider": "gpt-4",
                    "patient_id": 123,
                    "user_id": 1
                }
            }
        }