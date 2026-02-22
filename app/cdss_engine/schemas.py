# app/cdss_engine/schemas.py
"""
CDSS Request/Response Schemas - Updated with NoImageProvided handling
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
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
        


# # app/cdss_engine/schemas.py
# """
# CDSS Schemas - Updated with structured vision support and scientific confidence scoring
# """
# from pydantic import BaseModel, Field
# from typing import List, Optional, Literal, Union, Dict, Any
# from datetime import datetime

# # ============================================================================
# # PATIENT HISTORY
# # ============================================================================
# class PatientHistory(BaseModel):
#     """Patient history from database + current complaint."""
#     patient_id: Optional[str] = None
#     age: Optional[int] = None
#     gender: Optional[str] = None
#     chief_complaint: Optional[str] = None
#     medical_history: List[str] = Field(default_factory=list)
#     current_medications: List[str] = Field(default_factory=list)
#     allergies: List[str] = Field(default_factory=list)
#     previous_dental_work: Optional[str] = None
#     symptoms_duration: Optional[str] = None
#     pain_level: Optional[int] = Field(None, ge=0, le=10)

# # ============================================================================
# # VISION ANALYSIS - UPDATED WITH STRUCTURED SUPPORT
# # ============================================================================
# class ImageObservation(BaseModel):
#     """Radiograph analysis with structured and narrative components."""
    
#     # Structured findings (preferred)
#     structured_findings: Optional[Dict[str, Any]] = Field(
#         None,
#         description="Structured JSON pathology findings from vision model"
#     )
    
#     # Narrative components (backward compatibility)
#     raw_description: str = Field(..., description="Complete narrative radiographic analysis")
#     pathology_summary: str = Field(..., description="Focused pathology summary")
    
#     # Metadata
#     model_used: str = Field(..., description="Vision model used")
#     focused_tooth: Optional[str] = Field(None, description="Specific tooth that was analyzed")
    
#     # SCIENTIFIC CONFIDENCE SCORING
#     image_quality_score: float = Field(
#         default=0.5,
#         ge=0.0,
#         le=1.0,
#         description="Image technical quality (0-1, affects diagnostic confidence)"
#     )
#     diagnostic_confidence: float = Field(
#         default=0.5,
#         ge=0.0,
#         le=1.0,
#         description="Confidence in radiographic interpretation (0-1)"
#     )
#     overall_confidence: Literal["high", "medium", "low"] = Field(
#         default="medium",
#         description="Overall confidence category derived from scores"
#     )

# class NoImageProvided(BaseModel):
#     """Indicator that no image was provided."""
#     message: str = Field(default="No radiograph image was provided for this consultation")
#     image_required: bool = Field(default=False)

# # ============================================================================
# # KNOWLEDGE RETRIEVAL
# # ============================================================================
# class RetrievedKnowledge(BaseModel):
#     """Retrieved clinical guidelines chunk."""
#     content: str
#     pages: List[int]
#     relevance_score: Optional[float] = None
#     source: str = "Clinical Guidelines"

# # ============================================================================
# # CLINICAL RECOMMENDATION - UPDATED WITH SCIENTIFIC CONFIDENCE
# # ============================================================================
# class ClinicalRecommendation(BaseModel):
#     """Final structured clinical recommendation."""
#     diagnosis: str = Field(..., description="Primary clinical diagnosis")
#     differential_diagnoses: List[str] = Field(
#         default_factory=list,
#         description="Alternative diagnoses to consider"
#     )
#     recommended_management: str = Field(..., description="Treatment plan and next steps")
#     reference_pages: List[int] = Field(
#         default_factory=list,
#         description="Page numbers from guidelines (integers only)"
#     )
    
#     # SCIENTIFIC CONFIDENCE SCORING
#     confidence_score: float = Field(
#         default=0.5,
#         ge=0.0,
#         le=1.0,
#         description="Overall confidence score (0-1) based on multiple factors"
#     )
#     confidence_level: Literal["high", "medium", "low"] = Field(
#         default="medium",
#         description="Confidence category derived from score"
#     )
#     confidence_factors: Dict[str, float] = Field(
#         default_factory=dict,
#         description="Individual confidence factors that contributed to overall score"
#     )
    
#     llm_provider: str = Field(default="ollama", description="LLM used")

# # ============================================================================
# # CDSS REQUEST
# # ============================================================================
# class CDSSRequest(BaseModel):
#     """Request for CDSS recommendation."""
#     patient_id: int = Field(..., description="Patient database ID")
#     chief_complaint: str = Field(..., description="Main clinical complaint")
#     user_id: Optional[int] = Field(1, description="Requesting doctor ID")

# # ============================================================================
# # CDSS RESPONSE
# # ============================================================================
# class CDSSResponse(BaseModel):
#     """Complete CDSS response."""
    
#     recommendation: ClinicalRecommendation
#     image_observations: Union[ImageObservation, NoImageProvided] = Field(
#         ...,
#         description="Radiograph analysis or NoImageProvided message"
#     )
#     knowledge_sources: List[RetrievedKnowledge] = Field(default_factory=list)
#     reasoning_chain: str = Field(..., description="Step-by-step reasoning")
#     processing_metadata: dict = Field(default_factory=dict)
#     timestamp: datetime = Field(default_factory=datetime.utcnow)

# # ============================================================================
# # VISION-ONLY RESPONSE
# # ============================================================================
# class VisionAnalysisResponse(BaseModel):
#     """Response from vision analysis endpoint."""
    
#     # Structured findings
#     structured_findings: Optional[Dict[str, Any]] = Field(
#         None,
#         description="Structured pathology findings if available"
#     )
    
#     # Narrative components
#     detailed_description: str = Field(..., description="Complete analysis")
#     pathology_summary: str = Field(..., description="Key findings")
    
#     # Metadata
#     model_used: str = Field(..., description="Vision model used")
#     processing_time_ms: float = Field(..., description="Processing time")
#     focused_tooth: Optional[str] = Field(None, description="Focused tooth if specified")
    
#     # Scientific confidence
#     image_quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
#     diagnostic_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
#     confidence_level: Literal["high", "medium", "low"] = Field(default="medium")