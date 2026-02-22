# app/visionsystem/gemma3_client.py
"""
Gemma 3 Vision Client - Google's latest multimodal model via Ollama
"""
import base64
from io import BytesIO
from PIL import Image
import ollama
import logging
from config.visionconfig import vision_settings

logger = logging.getLogger(__name__)

class Gemma3VisionClient:
    """Gemma 3 client optimized for local dental radiograph analysis via Ollama."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_model_name(self):
        """Get Gemma 3 model name from settings."""
        return vision_settings.GEMMA3_MODEL
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 for Ollama."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def analyze_image(self, image: Image.Image, prompt: str) -> dict:
        """
        Analyze dental image using Gemma 3.
        """
        model_name = self._get_model_name()
        b64_image = self._encode_image(image)
        
        logger.info(f"🌟 Analyzing with Gemma 3 ({model_name}) via Ollama...")
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert dental radiologist. Provide precise, clinical analysis of dental X-rays."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64_image]
                    }
                ],
                options={
                    "temperature": vision_settings.VISION_TEMPERATURE,
                    "num_predict": vision_settings.VISION_MAX_TOKENS,
                }
            )
            
            result = response["message"]["content"]
            logger.info(f"✅ Gemma 3 analysis complete: {len(result)} chars")
            
            # Extract token counts
            input_tokens = response.get("prompt_eval_count")
            output_tokens = response.get("eval_count")
            logger.info(f"   Token usage: {input_tokens} prompt, {output_tokens} completion")

            return {
                "text": result,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            
        except Exception as e:
            logger.error(f"❌ Gemma 3 analysis failed: {e}")
            raise

# Global instance
gemma3_vision_client = Gemma3VisionClient()
