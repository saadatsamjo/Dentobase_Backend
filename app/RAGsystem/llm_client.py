# app/RAGsystem/llm_client.py
"""
LLM Client with STRUCTURED OUTPUT using Pydantic
Updated to properly extract page references from retrieved knowledge
"""
import logging
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from config.ragconfig import rag_settings

logger = logging.getLogger(__name__)


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

    class Config:
        json_schema_extra = {
            "example": {
                "diagnosis": "Irreversible pulpitis with periapical abscess, tooth #30",
                "differential_diagnoses": ["Acute apical periodontitis", "Cracked tooth syndrome"],
                "recommended_management": "1. Emergency treatment: Pulpectomy or extraction; 2. Pain management with NSAIDs (Ibuprofen 400mg TID); 3. Antibiotics if systemic involvement (Amoxicillin 500mg TID for 7 days); 4. Referral to endodontist for definitive root canal therapy; 5. Follow-up radiograph in 3-6 months.",
                "reference_pages": [100, 101, 352],
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

    def _extract_available_pages(self, retrieved_knowledge: str) -> List[int]:
        """
        Extract all page numbers from the retrieved knowledge context.
        This ensures we only reference pages that were actually retrieved.

        Args:
            retrieved_knowledge: The formatted knowledge context

        Returns:
            List of unique page numbers available in the retrieved knowledge
        """
        available_pages = set()

        # Match patterns like "Pages [385]" or "Pages 385" or "[1] [Pages 385, 386]"
        # More flexible pattern to catch various formats
        page_patterns = [
            r"Pages?\s*\[([0-9,\s]+)\]",  # Pages [385] or Page [385]
            r"Pages?\s*([0-9,\s]+)",  # Pages 385 or Pages 385, 386
            r"\[Pages?\s*([0-9,\s]+)\]",  # [Pages 385] or [Page 385]
        ]

        for pattern in page_patterns:
            matches = re.findall(pattern, retrieved_knowledge, re.IGNORECASE)
            for match in matches:
                # Extract individual page numbers from comma-separated strings
                page_nums = re.findall(r"\d+", match)
                available_pages.update(int(p) for p in page_nums)

        sorted_pages = sorted(list(available_pages))
        logger.info(f"📖 Available pages from retrieved knowledge: {sorted_pages}")
        return sorted_pages

    def generate_clinical_recommendation(
        self,
        patient_context: str,
        image_findings: str,
        retrieved_knowledge: str,
        query: str,
        knowledge_available: bool,
        available_pages: List[int] = None,  # NEW PARAMETER
    ) -> Dict[str, Any]:
        """
        Generate structured clinical recommendation.

        Args:
            patient_context: Patient demographics and history
            image_findings: Radiograph analysis results
            retrieved_knowledge: Retrieved guideline chunks
            query: Clinical question
            knowledge_available: Whether any knowledge chunks were retrieved
            available_pages: List of page numbers actually retrieved (optional, will be extracted if not provided)

        Returns:
            Dict matching ClinicalRecommendationOutput schema
        """
        provider = rag_settings.LLM_PROVIDER
        client_or_tuple = self._get_structured_client()

        # Extract available pages if not provided
        if available_pages is None and knowledge_available:
            available_pages = self._extract_available_pages(retrieved_knowledge)
        elif not knowledge_available:
            available_pages = []

        # Build context-aware prompt
        if knowledge_available and available_pages:
            available_pages_str = ", ".join(map(str, available_pages))
            guidelines_instruction = f"""You have clinical guidelines available from the following pages: {available_pages_str}
                        
                        CRITICAL INSTRUCTIONS FOR reference_pages:
                        1. ONLY cite page numbers from this list: {available_pages_str}
                        2. DO NOT make up or invent any page numbers
                        3. Only include pages whose content you actually used in your recommendation
                        4. If you don't use information from a specific page, don't include it in reference_pages
                        5. The reference_pages field must be a list of integers (e.g., [385, 386, 401])
                        6. DO NOT use text format like "Page 350" - only integers like 350"""
        elif knowledge_available:
            # Fallback if page extraction failed
            guidelines_instruction = """You have clinical guidelines available but page numbers could not be extracted.
                        CRITICAL: In reference_pages, use an empty list [] since specific page numbers are unavailable."""
        else:
            guidelines_instruction = """⚠️ WARNING: No clinical guidelines were retrieved from the knowledge base.
                        You must rely on general dental knowledge.
                        CRITICAL: In reference_pages, use an empty list [] since no guidelines were retrieved.
                        DO NOT make up page numbers."""

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
                    4. reference_pages: {"List of integer page numbers that you actually used (e.g., [385, 386])" if knowledge_available else "Empty list []"}

                    Provide your structured recommendation:"""

        logger.info(f"⚙️  Generating structured recommendation...")
        logger.info(f"   Provider: {provider}")
        logger.info(f"   Knowledge available: {knowledge_available}")
        logger.info(f"   Available pages: {available_pages}")

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

            # POST-PROCESSING: Validate and filter reference pages
            if result.get("reference_pages"):
                # Filter to only include pages that were actually available
                if available_pages:
                    filtered_pages = [p for p in result["reference_pages"] if p in available_pages]
                    if len(filtered_pages) != len(result["reference_pages"]):
                        logger.warning(
                            f"⚠️  Filtered hallucinated pages: {set(result['reference_pages']) - set(filtered_pages)}"
                        )
                    result["reference_pages"] = filtered_pages
                else:
                    # No pages available, clear any hallucinated references
                    logger.warning(
                        f"⚠️  Clearing hallucinated pages (no pages available): {result['reference_pages']}"
                    )
                    result["reference_pages"] = []

            logger.info(f"✅Structured output generated successfully")
            logger.info(f"   Diagnosis: {result['diagnosis'][:100]}...")
            logger.info(f"   Differential diagnoses: {len(result['differential_diagnoses'])} items")
            logger.info(f"   Management: {len(result['recommended_management'])} chars")
            logger.info(f"   Reference pages: {result['reference_pages']}")

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
                "reference_pages": [],
            }

    def _validate_output_structure(self, result: Dict) -> None:
        """Validate that output matches expected structure."""
        required_keys = [
            "diagnosis",
            "differential_diagnoses",
            "recommended_management",
            "reference_pages",
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

        if not isinstance(result["reference_pages"], list):
            raise ValueError(f"reference_pages must be list, got {type(result['reference_pages'])}")

        # Validate all items in reference_pages are integers
        for page in result["reference_pages"]:
            if not isinstance(page, int):
                raise ValueError(f"All items in reference_pages must be integers, got {type(page)}")

        logger.info(f"✅Output structure validated")

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
