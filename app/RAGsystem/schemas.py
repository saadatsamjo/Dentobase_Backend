# app/RAGsystem/schemas.py

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
    