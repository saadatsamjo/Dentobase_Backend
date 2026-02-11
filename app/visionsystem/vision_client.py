# app/visionsystem/vision_client.py
"""
Unified Vision Client
Routes to the appropriate vision model based on configuration
"""
import logging
from PIL import Image
from typing import Dict
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
    
    def analyze_dental_radiograph(self, image: Image.Image) -> Dict[str, str]:
        """
        Analyze a dental periapical radiograph for pathologies and anomalies.
        
        This is the PRIMARY method for CDSS integration.
        Uses specialized dental X-ray analysis prompts.
        
        Args:
            image: PIL Image of dental radiograph
            
        Returns:
            dict with:
                - detailed_description: Full clinical analysis
                - pathology_summary: Structured pathology findings
                - model: Which model was used
        """
        client = self._get_client()
        provider = vision_settings.VISION_MODEL_PROVIDER
        
        logger.info(f"Analyzing dental radiograph with {provider}...")
        
        # Use the clinical image analysis method
        result = client.analyze_clinical_image(image)
        
        # Ensure consistent output format
        return {
            "detailed_description": result.get("detailed_description", ""),
            "pathology_summary": result.get("region_findings", "No specific pathology detected"),
            "model": result.get("model", provider)
        }
    
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
            "dual_prompt": vision_settings.DUAL_PROMPT_ANALYSIS
        }

# Global instance
vision_client = VisionClient()