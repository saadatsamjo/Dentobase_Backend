# app/visionsystem/claude_vision_client.py
"""
Claude Vision Client - Optimized for Dental Radiographs
"""
from PIL import Image
import base64
from io import BytesIO
import os
import logging
from anthropic import Anthropic
from config.visionconfig import vision_settings

logger = logging.getLogger(__name__)

# Dental radiograph analysis prompt for Claude
DENTAL_XRAY_PROMPT = """You are analyzing a DENTAL PERIAPICAL RADIOGRAPH.

                TASK: Identify all pathologies and anomalies in this dental X-ray image.

                IMAGE CONTEXT:
                - This is a medical diagnostic image (periapical radiograph)
                - Shows tooth/teeth with roots and surrounding bone
                - Purpose: Detect abnormalities requiring clinical intervention

                REQUIRED ANALYSIS:

                1. IMAGE TYPE & QUALITY:
                - Confirm: periapical radiograph
                - Diagnostic quality assessment

                2. VISIBLE ANATOMY:
                - Tooth/teeth number (if identifiable)
                - Crown, root(s), pulp chamber
                - Alveolar bone, PDL, lamina dura

                3. PATHOLOGY IDENTIFICATION:

                DENTAL CARIES:
                - Dark areas in crown = decay
                - Specify: tooth, surface, depth
                
                PERIAPICAL PATHOLOGY:
                - Dark zone at root tip = abscess/cyst
                - Measure size, note characteristics
                
                BONE LOSS:
                - Reduced bone height around root
                - Type: horizontal/vertical
                - Severity: mild/moderate/severe
                
                RESTORATIONS:
                - Bright white = metal fillings
                - Quality assessment
                
                ROOT PATHOLOGY:
                - Resorption, fractures, calcifications

                4. RADIOGRAPHIC INTERPRETATION:
                - Radiolucent (dark) = less dense tissue, decay, infection
                - Radiopaque (bright) = dense tissue, metal, bone

                5. CLINICAL ASSESSMENT:
                - Primary diagnosis
                - Severity and urgency
                - Treatment implications

                BE SPECIFIC: Use tooth numbers, surface names, measurements when possible.
                USE DENTAL TERMINOLOGY: Proper clinical language.
                STATE CLEARLY: If no pathology detected, say so explicitly.

                Provide your detailed analysis:"""

class ClaudeVisionClient:
    """Claude Vision client for dental radiograph analysis."""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("CLAUDE_API_KEY")
            if not api_key:
                raise ValueError("CLAUDE_API_KEY not found in environment")
            self._client = Anthropic(api_key=api_key)
        return self._client
    
    def _get_model_name(self):
        """Get Claude model from RAG settings."""
        from config.ragconfig import rag_settings
        return rag_settings.CLAUDE_LLM_MODEL
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """Analyze dental radiograph using Claude Vision."""
        client = self._get_client()
        b64_image = self._encode_image(image)
        
        analysis_prompt = prompt or DENTAL_XRAY_PROMPT
        
        logger.info("Analyzing dental radiograph with Claude Vision...")
        
        try:
            response = client.messages.create(
                model=self._get_model_name(),
                max_tokens=vision_settings.VISION_MAX_TOKENS,
                temperature=vision_settings.VISION_TEMPERATURE,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": analysis_prompt
                            }
                        ]
                    }
                ]
            )
            
            result = response.content[0].text
            logger.info(f"Claude analysis complete: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Claude Vision analysis failed: {e}")
            raise
    
    def analyze_clinical_image(self, image: Image.Image) -> dict:
        """Comprehensive dental radiograph analysis."""
        # Single comprehensive analysis (Claude is good with complex prompts)
        detailed = self.analyze_image(image, DENTAL_XRAY_PROMPT)
        
        # Optional pathology-focused summary
        pathology = None
        if vision_settings.DUAL_PROMPT_ANALYSIS:
            try:
                pathology_prompt = """List only the pathological findings from this dental X-ray:
                    - Caries: present/absent, details if present
                    - Periapical lesions: present/absent, details if present  
                    - Bone loss: present/absent, details if present
                    - Other abnormalities: present/absent, details if present

                    Format as concise bullet points."""
                pathology = self.analyze_image(image, pathology_prompt)
            except Exception as e:
                logger.error(f"Pathology summary failed: {e}")
        
        return {
            "detailed_description": detailed,
            "region_findings": pathology or "See detailed description",
            "model": self._get_model_name()
        }

# Global instance
claude_vision_client = ClaudeVisionClient()