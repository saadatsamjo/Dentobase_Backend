# app/visionsystem/gemini_vision_client.py
"""
Google Gemini Vision Client - Updated for Google GenAI SDK
Supports: Gemini 2.0 Flash (Recommended), Gemini 1.5 Pro
"""
import logging
from PIL import Image
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class GeminiVisionClient:
    """Google Gemini multimodal vision client using modern GenAI SDK"""
    
    _instance = None
    _client = None
    _model_name = None
    _load_attempted = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _lazy_load_model(self):
        """Initialize Gemini client on first use"""
        if self._load_attempted:
            return
        self._load_attempted = True
        
        try:
            from config.visionconfig import vision_settings
            from google import genai
            
            api_key = vision_settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("⚠️ GEMINI_API_KEY not set - Gemini vision will fail")
                return
            
            # The new SDK uses a Client instance
            self._client = genai.Client(api_key=api_key)
            self._model_name = vision_settings.GEMINI_VISION_MODEL
            
            logger.info(f"✅ Gemini Client initialized: {self._model_name}")
        
        except ImportError:
            logger.error("❌ Google GenAI SDK not installed. Run: uv add google-genai")
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        """Analyze dental radiograph using Gemini 2.0/1.5."""
        from config.visionconfig import vision_settings
        from google.genai import types
        
        self._lazy_load_model()
        
        if not self._client:
            raise RuntimeError("Gemini client not initialized. Check API key.")
        
        logger.info(f"🌟 Analyzing with Gemini {self._model_name}...")
        
        try:
            # New SDK supports passing PIL objects directly in the contents list
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    temperature=vision_settings.VISION_TEMPERATURE,
                    max_output_tokens=vision_settings.VISION_MAX_TOKENS,
                )
            )
            
            if not response.text:
                raise RuntimeError("Gemini returned empty response or was blocked by safety filters.")
            
            result = response.text
            logger.info(f"✅ Gemini analysis complete: {len(result)} chars")
            return result
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Gemini analysis failed: {error_msg}")
            
            if "404" in error_msg:
                raise RuntimeError(f"Model {self._model_name} not found. Try updating to 'gemini-2.0-flash'.")
            raise RuntimeError(f"Gemini API error: {error_msg}")

# Global instance
gemini_vision_client = GeminiVisionClient()