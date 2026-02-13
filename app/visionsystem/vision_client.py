# app/visionsystem/vision_client.py
"""
Unified Vision Client - Fixed pathology summary extraction
"""
import logging
import re
from typing import Dict
from PIL import Image

from config.visionconfig import vision_settings

logger = logging.getLogger(__name__)


class VisionClient:
    """
    Unified interface to all vision models.
    Automatically routes to the correct model based on vision_settings.VISION_MODEL_PROVIDER
    """
    
    _instance = None
    _clients = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_client(self):
        """Get the appropriate vision client based on configuration."""
        provider = vision_settings.VISION_MODEL_PROVIDER
        
        # Lazy load clients only when needed
        if provider not in self._clients:
            if provider == "llava":
                from app.visionsystem.llava_client import llava_client
                self._clients["llava"] = llava_client
            elif provider == "gpt4v":
                from app.visionsystem.gpt4_client import gpt4v_client
                self._clients["gpt4v"] = gpt4v_client
            elif provider == "claude":
                from app.visionsystem.claude_vision_client import claude_vision_client
                self._clients["claude"] = claude_vision_client
            elif provider == "florence":
                from app.visionsystem.florence_client import florence_client
                self._clients["florence"] = florence_client
            else:
                raise ValueError(f"Unknown vision provider: {provider}")
        
        return self._clients[provider]
    
    def _extract_pathology_summary(self, detailed_description: str) -> str:
        """
        Extract the PATHOLOGY SUMMARY section from the detailed description.
        
        Args:
            detailed_description: Full analysis text
            
        Returns:
            Extracted pathology summary or original if not found
        """
        # Try to find the PATHOLOGY SUMMARY section
        patterns = [
            r'PATHOLOGY SUMMARY:?\s*\n(.*?)(?:\n\n|$)',  # Section with blank line after
            r'PATHOLOGY SUMMARY:?\s*\n(.*)',              # Section to end of text
        ]
        
        for pattern in patterns:
            match = re.search(pattern, detailed_description, re.DOTALL | re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                if summary and len(summary) > 10:  # Ensure we got meaningful content
                    logger.info(f"✅ Extracted pathology summary: {len(summary)} chars")
                    return summary
        
        # If no PATHOLOGY SUMMARY section found, try to extract structured findings
        logger.warning("⚠️  No PATHOLOGY SUMMARY section found, using full description")
        return detailed_description
    
    def analyze_dental_radiograph(
        self, 
        image: Image.Image,
        context: str = None,
        clinical_notes: str = None
    ) -> Dict[str, str]:
        """
        Analyze a dental periapical radiograph for pathologies and anomalies.
        
        This is the PRIMARY method for CDSS integration.
        Uses specialized dental X-ray analysis prompts with optional clinical context.
        
        Args:
            image: PIL Image of dental radiograph
            context: Optional clinical context (e.g., chief complaint, tooth numbers)
            clinical_notes: Optional relevant clinical notes
            
        Returns:
            dict with:
                - detailed_description: Full clinical analysis
                - pathology_summary: Extracted structured pathology findings
                - model: Which model was used
        """
        client = self._get_client()
        provider = vision_settings.VISION_MODEL_PROVIDER
        
        logger.info(f"Analyzing dental radiograph with {provider}...")
        if context:
            logger.info(f"📋 Clinical context provided: {context[:100]}...")
        if clinical_notes:
            logger.info(f"📝 Clinical notes provided: {len(clinical_notes)} chars")
        
        # Build enhanced prompt if context is provided
        if context or clinical_notes:
            enhanced_prompt = self._build_contextual_prompt(context, clinical_notes)
            logger.info(f"✅ Using context-enhanced analysis")
            
            # LOG THE ACTUAL PROMPT BEING SENT
            logger.info(f"\n{'─'*70}")
            logger.info(f"📤 PROMPT SENT TO VISION MODEL:")
            logger.info(f"{'─'*70}")
            logger.info(f"{enhanced_prompt[:500]}...")  # Show first 500 chars
            logger.info(f"{'─'*70}\n")
            
            # Use enhanced prompt with the client's analyze_image method
            result = client.analyze_image(image, enhanced_prompt)
            
            # For compatibility, wrap string response in expected dict format
            if isinstance(result, str):
                # Extract pathology summary from the detailed description
                pathology_summary = self._extract_pathology_summary(result)
                
                return {
                    "detailed_description": result,
                    "pathology_summary": pathology_summary,
                    "model": provider
                }
            else:
                return result
        else:
            # Use the standard clinical image analysis method
            result = client.analyze_clinical_image(image)
        
        # Ensure consistent output format
        pathology_summary = result.get("region_findings", "")
        if not pathology_summary or pathology_summary == "No specific pathology detected":
            # Try to extract from detailed_description
            pathology_summary = self._extract_pathology_summary(
                result.get("detailed_description", "")
            )
        
        return {
            "detailed_description": result.get("detailed_description", ""),
            "pathology_summary": pathology_summary,
            "model": result.get("model", provider)
        }
    
    def _build_contextual_prompt(self, context: str = None, clinical_notes: str = None) -> str:
        """
        Build an enhanced prompt that incorporates clinical context.
        
        Args:
            context: Clinical context (complaint, tooth numbers, etc.)
            clinical_notes: Relevant clinical notes
            
        Returns:
            Enhanced prompt string
        """
        base_prompt = """You are a dental radiologist analyzing a periapical X-ray image. 
                Provide a detailed clinical analysis of this dental radiograph, focusing on:

                1. ANATOMICAL STRUCTURES VISIBLE
                - Identify all teeth and structures in the image
                - Note the quality of the radiograph

                2. PATHOLOGY DETECTION - Systematically check for:
                a) DENTAL CARIES: Any radiolucent areas in tooth crowns indicating decay
                b) PERIAPICAL PATHOLOGY: Radiolucent areas at root apices (abscesses, granulomas, cysts)
                c) BONE LOSS: Horizontal or vertical bone loss around teeth
                d) PULPAL PATHOLOGY: Pulp stones, calcifications, or pulp chamber obliteration
                e) ROOT PATHOLOGY: Root resorption, fractures, or abnormal root morphology
                f) EXISTING DENTAL WORK: Fillings, crowns, root canal fillings - assess their integrity

                3. CLINICAL ASSESSMENT
                - Severity level (mild/moderate/severe)
                - Urgency level (routine/prompt/urgent/emergency)
                - Primary diagnosis based on radiographic findings

                After your detailed analysis, provide a PATHOLOGY SUMMARY in this exact format:

                PATHOLOGY SUMMARY:

                1. CARIES (Cavities):
                - Yes/No:
                    - [Location and details if yes]

                2. PERIAPICAL LESION/ABSCESS:
                - Yes/No:
                    - [Details if yes]

                3. BONE LOSS:
                - Yes/No:
                    - [Type and location if yes]

                4. ROOT CANAL TREATMENT:
                - Yes/No:
                    - [Assessment if yes]

                5. RESTORATIONS (Fillings/Crowns):
                - Yes/No:
                    - [Type and condition if yes]

                6. OTHER ABNORMALITIES:
                - [List any other findings]
                """
        
        # Add clinical context if provided
        if context or clinical_notes:
            context_section = "\n\nCLINICAL CONTEXT:\n"
            
            if context:
                context_section += f"Chief Complaint/Context: {context}\n"
            
            if clinical_notes:
                context_section += f"Clinical Notes: {clinical_notes}\n"
            
            context_section += "\nIMPORTANT: Incorporate this clinical information into your radiographic analysis. "
            context_section += "Pay special attention to areas mentioned in the clinical context. "
            context_section += "However, base your radiographic findings ONLY on what you can actually see in the X-ray.\n"
            
            base_prompt = context_section + base_prompt
        
        return base_prompt
    
    def analyze_image(self, image: Image.Image, custom_prompt: str = None) -> str:
        """
        Generic image analysis with optional custom prompt.
        
        For CDSS, prefer analyze_dental_radiograph() instead.
        
        Args:
            image: PIL Image
            custom_prompt: Optional custom analysis prompt
            
        Returns:
            str: Analysis description
        """
        client = self._get_client()
        
        if custom_prompt:
            return client.analyze_image(image, custom_prompt)
        else:
            # Use default clinical analysis
            result = client.analyze_clinical_image(image)
            return result.get("detailed_description", "")
    
    def get_model_info(self) -> Dict:
        """Get information about the currently configured vision model."""
        return {
            "provider": vision_settings.VISION_MODEL_PROVIDER,
            "model": self._get_client()._get_model_name() if hasattr(self._get_client(), '_get_model_name') else vision_settings.VISION_MODEL_PROVIDER,
            "dual_prompt": vision_settings.DUAL_PROMPT_ANALYSIS,
            "context_aware": True
        }

# Global instance
vision_client = VisionClient()