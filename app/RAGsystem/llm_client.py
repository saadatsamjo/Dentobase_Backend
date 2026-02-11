# app/RAGsystem/llm_client.py
"""
LLM Client with STRUCTURED OUTPUT using Pydantic
Works with ANY LLM provider and handles zero knowledge chunks gracefully
"""
import logging
from typing import Any, Dict

from pydantic import BaseModel, Field

from config.ragconfig import rag_settings

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC OUTPUT SCHEMA
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
    page_references: list[str] = Field(
        default_factory=list,
        description="Page numbers or chapters from clinical guidelines, or 'Based on general practice' if no guidelines",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "diagnosis": "Irreversible pulpitis with periapical abscess, tooth #30",
                "differential_diagnoses": ["Acute apical periodontitis", "Cracked tooth syndrome"],
                "recommended_management": "1. Emergency treatment: Pulpectomy or extraction; 2. Pain management with NSAIDs (Ibuprofen 400mg TID); 3. Antibiotics if systemic involvement (Amoxicillin 500mg TID for 7 days); 4. Referral to endodontist for definitive root canal therapy; 5. Follow-up radiograph in 3-6 months.",
                "page_references": ["Page 100", "Page 101"],
            }
        }


class LLMClient:
    """Unified LLM client with structured Pydantic output."""

    _instance = None
    _clients = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_structured_client(self):
        """
        Get LLM client with structured output.

        Uses .with_structured_output() for OpenAI/Anthropic
        Falls back to prompt-based JSON for Ollama
        """
        provider = rag_settings.LLM_PROVIDER

        logger.info(f"🤖 Initializing {provider} LLM with structured output")

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            base_client = ChatOpenAI(
                model=rag_settings.OPENAI_LLM_MODEL, temperature=rag_settings.LLM_TEMPERATURE
            )

            # Use native structured output
            return base_client.with_structured_output(
                ClinicalRecommendationOutput, method="json_mode"
            )

        elif provider == "claude":
            from langchain_anthropic import ChatAnthropic

            base_client = ChatAnthropic(
                model=rag_settings.CLAUDE_LLM_MODEL, temperature=rag_settings.LLM_TEMPERATURE
            )

            # Use native structured output
            return base_client.with_structured_output(ClinicalRecommendationOutput)

        elif provider == "ollama":
            # Ollama doesn't support with_structured_output reliably
            # Use PydanticOutputParser instead
            from langchain_core.output_parsers import PydanticOutputParser
            from langchain_ollama import ChatOllama

            base_client = ChatOllama(
                model=rag_settings.OLLAMA_LLM_MODEL,
                temperature=rag_settings.LLM_TEMPERATURE,
                format="json",
            )

            parser = PydanticOutputParser(pydantic_object=ClinicalRecommendationOutput)

            # Return tuple of (client, parser) for Ollama
            return (base_client, parser)

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def generate_clinical_recommendation(
        self,
        patient_context: str,
        image_findings: str,
        retrieved_knowledge: str,
        query: str,
        knowledge_available: bool,
    ) -> Dict[str, Any]:
        """
        Generate structured clinical recommendation.

        Args:
            patient_context: Patient demographics and history
            image_findings: Radiograph analysis results
            retrieved_knowledge: Retrieved guideline chunks
            query: Clinical question
            knowledge_available: Whether any knowledge chunks were retrieved

        Returns:
            Dict matching ClinicalRecommendationOutput schema
        """
        provider = rag_settings.LLM_PROVIDER
        client_or_tuple = self._get_structured_client()

        # Build context-aware prompt
        if knowledge_available:
            guidelines_instruction = """You have clinical guidelines available. Use them to support your recommendations.
                        CRITICAL: Cite actual page numbers from the guidelines in page_references."""
        else:
            guidelines_instruction = """⚠️ WARNING: No clinical guidelines were retrieved from the knowledge base.
                        You must rely on general dental knowledge.
                        CRITICAL: In page_references, do NOT cite specific pages. Instead use:
                        - "Based on general dental practice"
                        - "Standard clinical guidelines" 
                        - "General endodontic principles"
                        DO NOT make up page numbers like "Page 45" or "Chapter 7"."""

        # Construct prompt
        prompt = f"""You are a clinical decision support system for dentistry.

                    === PATIENT INFORMATION ===
                    {patient_context}

                    === RADIOGRAPHIC FINDINGS ===
                    {image_findings}

                    === CLINICAL GUIDELINES ===
                    {retrieved_knowledge}

                    === CLINICAL QUERY ===
                    {query}

                    === INSTRUCTIONS ===
                    {guidelines_instruction}

                    Based on the above information, provide a structured clinical recommendation.

                    REQUIREMENTS:
                    1. diagnosis: Primary diagnosis as a single descriptive string
                    2. differential_diagnoses: List of 2-3 alternative diagnoses
                    3. recommended_management: Complete treatment plan as ONE STRING with numbered steps
                    Format: "1. First step; 2. Second step; 3. Third step"
                    4. page_references: {"Actual page numbers from guidelines" if knowledge_available else "Generic references like 'Based on general practice'"}

                    Provide your structured recommendation:"""

        logger.info(f"⚙️  Generating structured recommendation...")
        logger.info(f"   Provider: {provider}")
        logger.info(f"   Knowledge available: {knowledge_available}")

        try:
            if provider == "ollama":
                # Ollama: use PydanticOutputParser
                client, parser = client_or_tuple

                # Add format instructions to prompt
                format_instructions = parser.get_format_instructions()
                full_prompt = f"{prompt}\n\n{format_instructions}"

                # Invoke
                response = client.invoke(full_prompt)

                # Parse with Pydantic
                parsed = parser.parse(response.content)
                result = parsed.dict()

            else:
                # OpenAI/Claude: use with_structured_output
                structured_client = client_or_tuple

                # Invoke directly - returns Pydantic object
                response = structured_client.invoke(prompt)
                result = response.dict()

            logger.info(f"✓ Structured output generated successfully")
            logger.info(f"   Diagnosis: {result['diagnosis'][:100]}...")
            logger.info(f"   Differential diagnoses: {len(result['differential_diagnoses'])} items")
            logger.info(f"   Management: {len(result['recommended_management'])} chars")
            logger.info(f"   Page references: {len(result['page_references'])} items")

            # Validate output types
            self._validate_output_structure(result)

            return result

        except Exception as e:
            logger.error(f"❌ Structured output generation failed: {e}", exc_info=True)

            # Return safe fallback
            return {
                "diagnosis": f"Error generating recommendation: {str(e)}",
                "differential_diagnoses": [],
                "recommended_management": "Unable to generate recommendation. Please consult supervising dentist.",
                "page_references": ["Error - no references available"],
            }

    def _validate_output_structure(self, result: Dict) -> None:
        """Validate that output matches expected structure."""
        required_keys = [
            "diagnosis",
            "differential_diagnoses",
            "recommended_management",
            "page_references",
        ]

        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key: {key}")

        # Type checks
        if not isinstance(result["diagnosis"], str):
            raise ValueError(f"diagnosis must be string, got {type(result['diagnosis'])}")

        if not isinstance(result["differential_diagnoses"], list):
            raise ValueError(
                f"differential_diagnoses must be list, got {type(result['differential_diagnoses'])}"
            )

        if not isinstance(result["recommended_management"], str):
            raise ValueError(
                f"recommended_management must be string, got {type(result['recommended_management'])}"
            )

        if not isinstance(result["page_references"], list):
            raise ValueError(f"page_references must be list, got {type(result['page_references'])}")

        logger.info(f"✓ Output structure validated")

    def get_model_info(self) -> Dict:
        """Get current LLM configuration."""
        return {
            "provider": rag_settings.LLM_PROVIDER,
            "model": rag_settings.current_llm_model,
            "temperature": rag_settings.LLM_TEMPERATURE,
            "structured_output": True,
            "parser": (
                "with_structured_output"
                if rag_settings.LLM_PROVIDER in ["openai", "claude"]
                else "PydanticOutputParser"
            ),
        }


# Global instance
llm_client = LLMClient()
