# app/RAGsystem/schemas.py

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ============================================================================
# PYDANTIC OUTPUT SCHEMAS
# ============================================================================
class ClinicalRecommendationOutput(BaseModel):
    """Structured output schema for LLM recommendations."""

    diagnosis: str = Field(..., description="Primary clinical diagnosis as a single string")
    differential_diagnoses: list[str] = Field(
        default_factory=list, description="List of alternative diagnoses to consider"
    )
    recommended_management: str = Field(
        ..., description="Complete treatment plan as a single string with numbered steps"
    )
    reference_pages: list[int] = Field(
        default_factory=list,
        description="Page numbers from clinical guidelines (integers only). Only include pages that were actually used from the retrieved knowledge. Do NOT invent page numbers.",
    )

    # class Config:
    #     json_schema_extra = {
    #         "example": {
    #             "diagnosis": "Irreversible pulpitis with periapical abscess, tooth #{tooth_number}",
    #             "differential_diagnoses": ["Acute apical periodontitis", "Cracked tooth syndrome"],
    #             "recommended_management": "1. Emergency treatment: Pulpectomy or extraction; 2. Pain management with NSAIDs (Ibuprofen 400mg TID); 3. Antibiotics if systemic involvement (Amoxicillin 500mg TID for 7 days); 4. Referral to endodontist for definitive root canal therapy; 5. Follow-up radiograph in 3-6 months.",
    #             "reference_pages": [100, 101, 352],
    #         }
    #     }


class QuestionPayload(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: Union[Dict[str, Any], str]  # Can be dict or string
    retrieval_strategy: str


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    success: bool


class PharmacologicalTreatment(BaseModel):
    """Standard pharmacological treatment structure"""

    analgesics: List[Dict[str, Any]] = Field(default_factory=list)
    antibiotics: List[Dict[str, Any]] = Field(default_factory=list)


class RecommendedManagement(BaseModel):
    """Standard management structure"""

    pharmacological: PharmacologicalTreatment
    non_pharmacological: List[Dict[str, Any]] = Field(default_factory=list)
    follow_up: str = "Not specified"


class StandardizedRAGAnswer(BaseModel):
    """Enforced standard structure for ALL RAG responses"""

    diagnosis: str = Field(..., description="Primary diagnosis")
    differential_diagnoses: List[str] = Field(
        default_factory=list, description="Alternative diagnoses to consider"
    )
    recommended_management: RecommendedManagement
    precautions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Contraindications and warnings"
    )
    reference_pages: List[int] = Field(
        default_factory=list, description="Page numbers from clinical guidelines (REQUIRED)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "diagnosis": "Periapical abscess, tooth #47",
                "differential_diagnoses": [
                    "Acute apical periodontitis",
                    "Symptomatic irreversible pulpitis",
                ],
                "recommended_management": {
                    "pharmacological": {
                        "analgesics": [
                            {
                                "name": "Ibuprofen",
                                "dose": "400mg 8 hourly",
                                "reference_page": 44,
                                "type": "analgesic",
                            }
                        ],
                        "antibiotics": [
                            {
                                "name": "Amoxicillin",
                                "dose": "500mg 8 hourly for 5-7 days",
                                "reference_page": 45,
                                "type": "antibiotic",
                            }
                        ],
                    },
                    "non_pharmacological": [
                        {
                            "description": "Extraction of offending tooth under local anesthesia",
                            "category": "extraction",
                            "reference_page": 43,
                        },
                        {
                            "description": "Establish drainage with incision",
                            "category": "drainage",
                            "reference_page": 43,
                        },
                    ],
                    "follow_up": "Review in 24-48 hours if symptoms persist",
                },
                "precautions": [
                    {
                        "condition": "Pregnancy",
                        "note": "Adjust antibiotic - avoid tetracyclines",
                        "reference_page": 47,
                    }
                ],
                "reference_pages": [43, 44, 45, 47],
            }
        }
