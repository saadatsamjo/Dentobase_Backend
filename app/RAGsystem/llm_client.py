# app/RAGsystem/llm_client.py
"""
LLM Client - FIXED with simple text parsing (no complex Pydantic)
This version uses regex-based extraction instead of structured output for reliability with Ollama
"""
import logging
import re
from typing import Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from config.ragconfig import rag_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client with simple text-based output parsing (reliable with all providers)"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def generate_clinical_recommendation(
        self,
        patient_context: str,
        image_findings: str,
        retrieved_knowledge: str,
        query: str,
        knowledge_available: bool,
        available_pages: List[int] = None,
    ) -> Dict:
        """
        Generate clinical recommendation using SIMPLE text format.
        
        This approach is MORE RELIABLE than JSON/Pydantic with local LLMs.
        """
        provider = rag_settings.LLM_PROVIDER
        logger.info(f"🤖 Generating recommendation with {provider} - Model: {rag_settings.current_llm_model}")
        
        # Extract pages from retrieved knowledge if not provided
        if available_pages is None and knowledge_available:
            available_pages = self._extract_available_pages(retrieved_knowledge)
        elif not knowledge_available:
            available_pages = []
        
        # Build SIMPLE prompt (no JSON schema)
        prompt = self._build_simple_prompt(
            patient_context,
            image_findings,
            retrieved_knowledge,
            query,
            available_pages
        )
        
        # Get LLM client
        llm = self._get_llm_client()
        
        try:
            # Invoke LLM
            logger.info(f"⚙️  Calling LLM...")
            response = llm.invoke(prompt)
            
            # Parse simple text response
            result = self._parse_simple_response(response.content, available_pages)
            
            logger.info(f"✅ Generated recommendation successfully")
            logger.info(f"   Diagnosis: {result['diagnosis'][:80]}...")
            logger.info(f"   Reference pages: {result['reference_pages']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_response()
    
    def _get_llm_client(self):
        """Get simple LLM client (no structured output)"""
        provider = rag_settings.LLM_PROVIDER
        
        if provider == "ollama":
            return ChatOllama(
                model=rag_settings.OLLAMA_LLM_MODEL,
                temperature=rag_settings.LLM_TEMPERATURE
            )
        elif provider == "openai":
            return ChatOpenAI(
                model=rag_settings.OPENAI_LLM_MODEL,
                temperature=rag_settings.LLM_TEMPERATURE
            )
        elif provider == "claude":
            return ChatAnthropic(
                model=rag_settings.CLAUDE_LLM_MODEL,
                temperature=rag_settings.LLM_TEMPERATURE
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _build_simple_prompt(
        self,
        patient_context: str,
        image_findings: str,
        retrieved_knowledge: str,
        query: str,
        available_pages: List[int]
    ) -> str:
        """Build SIMPLE text prompt (much more reliable than JSON)"""
        
        pages_str = ", ".join(map(str, available_pages)) if available_pages else "None"
        
        prompt = f"""You are a clinical decision support system for dentistry.

                === PATIENT INFORMATION ===
                {patient_context}

                === RADIOGRAPHIC FINDINGS ===
                {image_findings}

                === CLINICAL GUIDELINES ===
                {retrieved_knowledge}

                === AVAILABLE PAGES ===
                You may ONLY reference these page numbers: {pages_str}
                DO NOT invent or make up any page numbers not in this list.

                === CLINICAL QUERY ===
                {query}

                === CRITICAL INSTRUCTIONS ===
                1. EXTRACT the tooth number from the patient context or clinical notes
                2. Your diagnosis MUST explicitly mention that specific tooth number
                3. Use ONLY page numbers from the available list above
                4. Format your response EXACTLY as shown below (do not add extra sections)

                === RESPONSE FORMAT ===
                Use this EXACT structure (include the section headers):

                DIAGNOSIS: [Write complete diagnosis here, MUST include tooth number]

                DIFFERENTIAL:
                - [Alternative diagnosis 1]
                - [Alternative diagnosis 2]

                MANAGEMENT:
                1. [First treatment step]
                2. [Second treatment step]  
                3. [Third treatment step]
                4. [Fourth treatment step if needed]

                PAGES: [comma-separated page numbers only, e.g., 385, 388, 401]

                Now provide your clinical recommendation:"""
        
        return prompt
    
    def _parse_simple_response(self, text: str, available_pages: List[int]) -> Dict:
        """
        Parse simple text format using regex (deterministic, no LLM involved).
        This is MUCH more reliable than JSON parsing with local LLMs.
        """
        logger.info(f"📥 Parsing response ({len(text)} chars)...")
        
        # Extract diagnosis
        diag_match = re.search(
            r'DIAGNOSIS:\s*(.+?)(?:\n\n|DIFFERENTIAL:|$)', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        diagnosis = diag_match.group(1).strip() if diag_match else "Unable to generate diagnosis"
        
        # Extract differentials
        diff_match = re.search(
            r'DIFFERENTIAL:\s*(.+?)(?:\n\n|MANAGEMENT:|$)', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        differentials = []
        if diff_match:
            diff_text = diff_match.group(1)
            # Extract lines starting with dash or number
            diff_lines = re.findall(r'[-•]\s*(.+?)(?:\n|$)', diff_text)
            differentials = [d.strip() for d in diff_lines if d.strip()][:3]  # Max 3
        
        # Extract management
        mgmt_match = re.search(
            r'MANAGEMENT:\s*(.+?)(?:\n\n|PAGES:|$)', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        management = mgmt_match.group(1).strip() if mgmt_match else "Consult supervising dentist for treatment plan."
        
        # Extract pages
        pages_match = re.search(
            r'PAGES:\s*(.+?)(?:\n|$)', 
            text, 
            re.IGNORECASE
        )
        pages = []
        if pages_match:
            # Extract all numbers
            page_nums = re.findall(r'\d+', pages_match.group(1))
            # Filter to only available pages (validation)
            pages = [int(p) for p in page_nums if int(p) in available_pages]
        
        logger.info(f"✅ Parsed successfully:")
        logger.info(f"   Diagnosis: {len(diagnosis)} chars")
        logger.info(f"   Differentials: {len(differentials)}")
        logger.info(f"   Management: {len(management)} chars")
        logger.info(f"   Pages: {pages}")
        
        return {
            "diagnosis": diagnosis,
            "differential_diagnoses": differentials,
            "recommended_management": management,
            "reference_pages": pages
        }
    
    def _extract_available_pages(self, retrieved_knowledge: str) -> List[int]:
        """
        Extract all page numbers from retrieved knowledge context.
        Ensures we only reference pages that were actually retrieved.
        """
        available_pages = set()
        
        # Match patterns like "Pages [385]" or "[1] Pages [385, 386]"
        page_patterns = [
            r'Pages?\s*\[([0-9,\s]+)\]',  # Pages [385]
            r'Pages?\s*([0-9,\s]+)',      # Pages 385, 386
            r'\[Pages?\s*([0-9,\s]+)\]',  # [Pages 385]
        ]
        
        for pattern in page_patterns:
            matches = re.findall(pattern, retrieved_knowledge, re.IGNORECASE)
            for match in matches:
                page_nums = re.findall(r'\d+', match)
                available_pages.update(int(p) for p in page_nums)
        
        sorted_pages = sorted(list(available_pages))
        logger.info(f"📖 Extracted pages from knowledge: {sorted_pages}")
        return sorted_pages
    
    def _fallback_response(self) -> Dict:
        """Fallback response if generation completely fails"""
        return {
            "diagnosis": "Unable to generate recommendation - system error. Please consult supervising dentist.",
            "differential_diagnoses": [],
            "recommended_management": "1. Comprehensive clinical examination; 2. Additional diagnostic imaging if needed; 3. Consult supervising dentist for diagnosis and treatment planning.",
            "reference_pages": []
        }
    
    def get_model_info(self) -> Dict:
        """Get current LLM configuration"""
        return {
            "provider": rag_settings.LLM_PROVIDER,
            "model": rag_settings.current_llm_model,
            "temperature": rag_settings.LLM_TEMPERATURE,
            "structured_output": False,  # Using simple text parsing
            "parser": "regex",
            "reliable": True  # Text parsing is much more reliable
        }


# Global instance
llm_client = LLMClient()

