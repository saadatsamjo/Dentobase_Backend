# app/cdss_engine/schemas.py
"""
CDSS Request/Response Schemas - Refactored
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
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
    raw_description: str = Field(..., description="Full clinical analysis from vision model")
    pathology_summary: str = Field(..., description="Focused pathology findings")
    confidence: Literal["high", "medium", "low"] = "medium"
    model_used: str = Field(..., description="Which vision model was used")

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
    page_references: List[str] = Field(
        default_factory=list,
        description="Page numbers or chapters from guidelines"
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
    image_observations: Optional[ImageObservation] = Field(
        None,
        description="Radiograph analysis if image was provided"
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
                    "page_references": ["Page 45", "Chapter 7: Endodontic Treatment"],
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

# ============================================================================
# VISION-ONLY RESPONSE (for /analyze_image endpoint)
# ============================================================================
class VisionAnalysisResponse(BaseModel):
    """Response from vision analysis endpoint."""
    detailed_description: str = Field(..., description="Complete radiograph analysis")
    pathology_summary: str = Field(..., description="Key pathological findings")
    model_used: str = Field(..., description="Vision model used")
    processing_time_ms: float = Field(..., description="Analysis time in milliseconds")
    confidence: Literal["high", "medium", "low"] = "medium"
    
    class Config:
        json_schema_extra = {
            "example": {
                "detailed_description": "Periapical radiograph of tooth #19 shows...",
                "pathology_summary": "Mesial caries, mild bone loss",
                "model_used": "llava:13b",
                "processing_time_ms": 3420.5,
                "confidence": "high"
            }
        }