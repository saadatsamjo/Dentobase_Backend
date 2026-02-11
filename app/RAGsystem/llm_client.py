# app/RAGsystem/llm_client.py
"""
Unified LLM Client with IMPROVED PROMPT and validation handling
"""
import logging
import json
from typing import Dict, Any
from config.ragconfig import rag_settings

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified LLM interface with robust JSON parsing."""
    
    _instance = None
    _clients = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_client(self):
        """Get appropriate LLM client based on configuration."""
        provider = rag_settings.LLM_PROVIDER
        
        if provider not in self._clients:
            if provider == "ollama":
                from langchain_ollama import ChatOllama
                self._clients["ollama"] = ChatOllama(
                    model=rag_settings.OLLAMA_LLM_MODEL,
                    temperature=rag_settings.LLM_TEMPERATURE,
                    format="json" if rag_settings.FORCE_JSON_OUTPUT else None
                )
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                self._clients["openai"] = ChatOpenAI(
                    model=rag_settings.OPENAI_LLM_MODEL,
                    temperature=rag_settings.LLM_TEMPERATURE,
                    model_kwargs={"response_format": {"type": "json_object"}} if rag_settings.FORCE_JSON_OUTPUT else {}
                )
            elif provider == "claude":
                from langchain_anthropic import ChatAnthropic
                self._clients["claude"] = ChatAnthropic(
                    model=rag_settings.CLAUDE_LLM_MODEL,
                    temperature=rag_settings.LLM_TEMPERATURE
                )
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")
        
        return self._clients[provider]
    
    def generate_clinical_recommendation(
        self,
        patient_context: str,
        image_findings: str,
        retrieved_knowledge: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Generate structured clinical recommendation.
        
        IMPROVED PROMPT: Explicitly requests string format, not lists.
        """
        client = self._get_client()
        
        # IMPROVED PROMPT - Clear about data types
        fusion_prompt = f"""You are a clinical decision support system for dentistry.

=== PATIENT INFORMATION ===
{patient_context}

=== RADIOGRAPHIC FINDINGS ===
{image_findings}

=== CLINICAL GUIDELINES (Retrieved from Knowledge Base) ===
{retrieved_knowledge}

=== CLINICAL QUERY ===
{query}

=== YOUR TASK ===
Based on the patient information, radiographic findings (if provided), and clinical guidelines above, provide a structured clinical recommendation.

=== OUTPUT FORMAT ===
Return ONLY valid JSON with this EXACT structure:

{{
    "diagnosis": "Primary clinical diagnosis as a single string. Example: Irreversible pulpitis with periapical abscess, tooth #30",
    
    "differential_diagnoses": [
        "Alternative diagnosis 1",
        "Alternative diagnosis 2"
    ],
    
    "recommended_management": "Complete treatment plan as a SINGLE STRING (NOT a list). Include all steps in one paragraph separated by semicolons or numbered inline. Example: 1. Emergency treatment: Pulpectomy or extraction; 2. Pain management with NSAIDs (Ibuprofen 400mg); 3. Antibiotics if systemic involvement; 4. Follow-up in 3-6 months.",
    
    "page_references": [
        "Page 45",
        "Page 52",
        "Chapter 7: Endodontics"
    ]
}}

=== CRITICAL RULES ===
1. "diagnosis" MUST be a STRING (not array)
2. "differential_diagnoses" MUST be an ARRAY of strings
3. "recommended_management" MUST be a SINGLE STRING (NOT an array of objects)
4. "page_references" MUST be an ARRAY of strings citing actual page numbers from the guidelines
5. Base recommendations ONLY on provided guidelines
6. If no guidelines retrieved, use general dental knowledge but note this
7. Use proper dental terminology
8. Be specific and actionable

Generate the JSON response now:"""
        
        logger.info(f"🤖 Generating recommendation with {rag_settings.LLM_PROVIDER}...")
        logger.info(f"   Temperature: {rag_settings.LLM_TEMPERATURE}")
        logger.info(f"   JSON mode: {rag_settings.FORCE_JSON_OUTPUT}")
        
        try:
            response = client.invoke(fusion_prompt)
            
            # Extract content
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            logger.info(f"✓ LLM raw response received ({len(content)} chars)")
            logger.info(f"   First 200 chars: {content[:200]}...")
            
            # Parse JSON
            result = self._parse_json_response(content)
            
            # VALIDATE AND FIX structure
            result = self._validate_and_fix_structure(result)
            
            logger.info(f"✓ Recommendation parsed successfully")
            logger.info(f"   Diagnosis: {result.get('diagnosis', 'N/A')[:100]}...")
            logger.info(f"   Management type: {type(result.get('recommended_management'))}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}", exc_info=True)
            raise
    
    def _parse_json_response(self, content: str) -> Dict:
        """Parse JSON from LLM response with fallbacks."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass
        
        # Try extracting from triple backticks
        if "```" in content:
            try:
                json_str = content.split("```")[1].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass
        
        # Last resort: find JSON-like content
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass
        
        raise ValueError(f"Could not parse JSON from response: {content[:500]}")
    
    def _validate_and_fix_structure(self, result: Dict) -> Dict:
        """
        Validate and fix the response structure.
        
        CRITICAL: Fixes the 'recommended_management' being a list issue.
        """
        logger.info(f"🔍 Validating response structure...")
        
        # Fix diagnosis (must be string)
        if "diagnosis" not in result or not result["diagnosis"]:
            result["diagnosis"] = "Unable to determine diagnosis from provided information"
        elif isinstance(result["diagnosis"], list):
            result["diagnosis"] = "; ".join(result["diagnosis"])
        
        # Fix differential_diagnoses (must be list)
        if "differential_diagnoses" not in result:
            result["differential_diagnoses"] = []
        elif not isinstance(result["differential_diagnoses"], list):
            result["differential_diagnoses"] = [str(result["differential_diagnoses"])]
        
        # FIX recommended_management (MUST BE STRING, NOT LIST!)
        if "recommended_management" not in result:
            result["recommended_management"] = "Consult with supervising dentist for treatment planning"
        elif isinstance(result["recommended_management"], list):
            logger.warning(f"⚠️  recommended_management is a list, converting to string...")
            
            # Convert list of objects/strings to single string
            management_parts = []
            for i, item in enumerate(result["recommended_management"], 1):
                if isinstance(item, dict):
                    # Item is an object like {"step": "...", "description": "..."}
                    step = item.get("step", item.get("description", item.get("name", str(item))))
                    management_parts.append(f"{i}. {step}")
                elif isinstance(item, str):
                    management_parts.append(f"{i}. {item}")
                else:
                    management_parts.append(f"{i}. {str(item)}")
            
            result["recommended_management"] = "; ".join(management_parts)
            logger.info(f"✓ Fixed: Converted list to string ({len(result['recommended_management'])} chars)")
        
        # Fix page_references (must be list)
        if "page_references" not in result:
            result["page_references"] = []
        elif not isinstance(result["page_references"], list):
            result["page_references"] = [str(result["page_references"])]
        
        # Log final structure
        logger.info(f"✓ Validation complete:")
        logger.info(f"   diagnosis: {type(result['diagnosis']).__name__}")
        logger.info(f"   differential_diagnoses: {type(result['differential_diagnoses']).__name__} ({len(result['differential_diagnoses'])} items)")
        logger.info(f"   recommended_management: {type(result['recommended_management']).__name__}")
        logger.info(f"   page_references: {type(result['page_references']).__name__} ({len(result['page_references'])} items)")
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about currently configured LLM."""
        return {
            "provider": rag_settings.LLM_PROVIDER,
            "model": rag_settings.current_llm_model,
            "temperature": rag_settings.LLM_TEMPERATURE,
            "json_mode": rag_settings.FORCE_JSON_OUTPUT
        }

# Global instance
llm_client = LLMClient()