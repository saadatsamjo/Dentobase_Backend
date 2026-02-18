# app/visionsystem/vision_schemas.py
"""
Structured schemas for vision model outputs to ensure consistent, parseable results
"""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class VisionAnalysisResponse(BaseModel):
    """Response from vision analysis endpoint."""
    
    # Structured findings
    structured_findings: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured pathology findings if available"
    )
    
    # Narrative components
    detailed_description: str = Field(..., description="Complete analysis")
    pathology_summary: str = Field(..., description="Key findings")
    
    # Metadata
    model_used: str = Field(..., description="Vision model used")
    processing_time_ms: float = Field(..., description="Processing time")
    focused_tooth: Optional[str] = Field(None, description="Focused tooth if specified")
    
    # Scientific confidence
    image_quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    diagnostic_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_level: Literal["high", "medium", "low"] = Field(default="medium")
    

class CariesFinding(BaseModel):
    """Structured caries finding"""
    present: bool = Field(..., description="Whether caries are present")
    location: Optional[str] = Field(None, description="Tooth number and surface (e.g., '23 mesial')")
    severity: Optional[Literal["early", "moderate", "deep"]] = Field(None)
    notes: Optional[str] = Field(None, description="Additional details")


class PeriapicalFinding(BaseModel):
    """Structured periapical pathology finding"""
    present: bool = Field(..., description="Whether periapical lesion/abscess present")
    location: Optional[str] = Field(None, description="Tooth number (e.g., '23')")
    size_mm: Optional[float] = Field(None, description="Approximate size in mm if measurable")
    characteristics: Optional[str] = Field(None, description="Well-defined, diffuse, etc.")
    notes: Optional[str] = Field(None)


class BoneLossFinding(BaseModel):
    """Structured bone loss finding"""
    present: bool = Field(..., description="Whether bone loss is present")
    type: Optional[Literal["horizontal", "vertical", "mixed"]] = Field(None)
    location: Optional[str] = Field(None, description="Which teeth affected")
    severity: Optional[Literal["mild", "moderate", "severe"]] = Field(None)
    notes: Optional[str] = Field(None)


class RootCanalTreatment(BaseModel):
    """Structured root canal treatment assessment"""
    present: bool = Field(..., description="Whether RCT is present")
    location: Optional[str] = Field(None, description="Tooth number")
    quality: Optional[Literal["adequate", "short", "overfilled", "underfilled"]] = Field(None)
    notes: Optional[str] = Field(None)


class RestorationFinding(BaseModel):
    """Structured restoration finding"""
    present: bool = Field(..., description="Whether restorations are present")
    type: Optional[str] = Field(None, description="Amalgam, composite, crown, etc.")
    location: Optional[str] = Field(None, description="Tooth number and surface")
    condition: Optional[Literal["intact", "defective", "overhanging", "recurrent decay"]] = Field(None)
    notes: Optional[str] = Field(None)


class OtherAbnormality(BaseModel):
    """Any other abnormal finding"""
    description: str = Field(..., description="Description of the abnormality")
    location: Optional[str] = Field(None)
    clinical_significance: Optional[Literal["urgent", "prompt", "routine", "monitor"]] = Field(None)


class StructuredPathologyFindings(BaseModel):
    """Complete structured pathology findings from radiograph"""
    
    # Anatomical assessment
    teeth_visible: List[str] = Field(
        default_factory=list,
        description="List of tooth numbers visible (e.g., ['23', '24', '25'])"
    )
    image_quality: Literal["excellent", "good", "adequate", "poor"] = Field(
        ...,
        description="Overall radiograph quality"
    )
    
    # Pathology findings
    caries: CariesFinding = Field(..., description="Caries findings")
    periapical_pathology: PeriapicalFinding = Field(..., description="Periapical findings")
    bone_loss: BoneLossFinding = Field(..., description="Bone loss findings")
    root_canal_treatment: RootCanalTreatment = Field(..., description="RCT findings")
    restorations: RestorationFinding = Field(..., description="Restoration findings")
    other_abnormalities: List[OtherAbnormality] = Field(
        default_factory=list,
        description="Any other abnormal findings"
    )
    
    # Clinical assessment
    primary_radiographic_finding: str = Field(
        ...,
        description="Main finding based on radiograph alone"
    )
    severity: Literal["normal", "mild", "moderate", "severe"] = Field(
        ...,
        description="Overall severity based on radiographic findings"
    )
    urgency: Literal["emergency", "urgent", "prompt", "routine"] = Field(
        ...,
        description="Recommended urgency level"
    )
    
    # Confidence scoring
    image_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score for image quality (affects diagnostic confidence)"
    )
    diagnostic_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score for confidence in the radiographic interpretation"
    )
    interpretation_notes: Optional[str] = Field(
        None,
        description="Any limitations or areas of uncertainty in interpretation"
    )


class VisionAnalysisResult(BaseModel):
    """Complete vision analysis result with both narrative and structured data"""
    
    # Narrative analysis (for context/review)
    narrative_analysis: str = Field(
        ...,
        description="Full narrative radiographic interpretation"
    )
    
    # Structured findings (for programmatic use)
    structured_findings: StructuredPathologyFindings = Field(
        ...,
        description="Structured, parseable pathology findings"
    )
    
    # Clinical context integration
    clinical_context_considered: bool = Field(
        default=False,
        description="Whether clinical context was provided and considered"
    )
    focused_tooth_number: Optional[str] = Field(
        None,
        description="Specific tooth number that was focus of analysis if provided"
    )
    
    # Model metadata
    model_used: str = Field(..., description="Vision model used for analysis")