# app/visionsystem/groq_vision_client.py
"""
Groq Vision Client - Ultra-fast vision inference using LPU
Supports: Llama 3.2 90B Vision, 11B Vision
Speed: 5-20x faster than GPU-based providers
Cost: ~$0.001 per image (100x cheaper than GPT-4V)
"""
import logging
import base64
from io import BytesIO
from PIL import Image
from typing import Dict

logger = logging.getLogger(__name__)


class GroqVisionClient:
    """Groq LPU-accelerated vision model client"""
    
    _instance = None
    _client = None
    _load_attempted = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _lazy_load_client(self):
        """Initialize Groq client on first use"""
        if self._load_attempted:
            return
        self._load_attempted = True
        
        try:
            from config.visionconfig import vision_settings
            import os
            
            api_key = vision_settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("⚠️  GROQ_API_KEY not set - Groq vision will fail")
                return
            
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info(f"✅ Groq client initialized with model: {vision_settings.GROQ_VISION_MODEL}")
        
        except ImportError:
            logger.error("❌ Groq SDK not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"❌ Groq client initialization failed: {e}")
    
    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        """
        Analyze dental radiograph using Groq vision model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for analysis
        
        Returns:
            str: Model response text
        """
        from config.visionconfig import vision_settings
        
        self._lazy_load_client()
        
        if not self._client:
            raise RuntimeError("Groq client not initialized. Check API key.")
        
        logger.info(f"🚀 Analyzing with Groq {vision_settings.GROQ_VISION_MODEL}...")
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        try:
            # Groq uses OpenAI-compatible API
            response = self._client.chat.completions.create(
                model=vision_settings.GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=vision_settings.VISION_TEMPERATURE,
                max_tokens=vision_settings.VISION_MAX_TOKENS,
            )
            
            result = response.choices[0].message.content
            logger.info(f"✅ Groq analysis complete: {len(result)} chars")
            return result
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Groq analysis failed: {error_msg}")
            
            # Handle specific errors
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                raise RuntimeError(f"Groq API quota exceeded: {error_msg}")
            elif "auth" in error_msg.lower() or "key" in error_msg.lower():
                raise RuntimeError(f"Groq API authentication failed: {error_msg}")
            else:
                raise RuntimeError(f"Groq analysis error: {error_msg}")


# Global instance
groq_vision_client = GroqVisionClient()