# app/visionsystem/gemini_vision_client.py
"""
Google Gemini Vision Client - UPDATED for new google.genai SDK
"""
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class GeminiVisionClient:
    """Google Gemini client using NEW SDK (google.genai)"""

    _instance = None
    _client = None
    _model_id = None
    _load_attempted = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _lazy_load_client(self):
        """Initialize with NEW google.genai SDK"""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            import os

            from config.visionconfig import vision_settings

            api_key = vision_settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("⚠️  GEMINI_API_KEY not set")
                return

            # NEW SDK import
            from google import genai

            self._client = genai.Client(api_key=api_key)
            self._model_id = vision_settings.GEMINI_VISION_MODEL

            logger.info(f"✅ Gemini client initialized: {self._model_id}")

        except ImportError:
            logger.error("❌ google-genai not installed. Run: pip install google-genai")
        except Exception as e:
            logger.error(f"❌ Gemini failed: {e}")

    def analyze_image(self, image: Image.Image, prompt: str) -> dict:
        """Analyze with NEW SDK"""
        from config.visionconfig import vision_settings

        self._lazy_load_client()

        if not self._client:
            raise RuntimeError("Gemini client not initialized")

        logger.info(f"🌟 Analyzing with Gemini {self._model_id}...")

        try:
            import base64
            from io import BytesIO

            # Convert PIL to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # NEW SDK API
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=[prompt, {"inline_data": {"mime_type": "image/png", "data": img_base64}}],
            )

            result = response.text
            usage = response.usage_metadata
            logger.info(f"✅ Gemini complete: {len(result)} chars")
            if usage:
                logger.info(f"   Token usage: {usage.prompt_token_count} prompt, {usage.candidates_token_count} completion")
                return {
                    "text": result,
                    "input_tokens": usage.prompt_token_count,
                    "output_tokens": usage.candidates_token_count,
                }
            else:
                return {"text": result, "input_tokens": None, "output_tokens": len(result.split())}

        except Exception as e:
            logger.error(f"❌ Gemini failed: {e}")
            raise RuntimeError(f"Gemini error: {e}")


gemini_vision_client = GeminiVisionClient()


# # app/visionsystem/gemini_vision_client.py

# """
# Google Gemini Vision Client - Updated for Google GenAI SDK
# Supports: Gemini 2.0 Flash (Recommended), Gemini 1.5 Pro
# """
# import logging
# from PIL import Image
# from typing import Dict, Optional
# import os

# logger = logging.getLogger(__name__)

# class GeminiVisionClient:
#     """Google Gemini multimodal vision client using modern GenAI SDK"""

#     _instance = None
#     _client = None
#     _model_name = None
#     _load_attempted = False

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def _lazy_load_model(self):
#         """Initialize Gemini client on first use"""
#         if self._load_attempted:
#             return
#         self._load_attempted = True

#         try:
#             from config.visionconfig import vision_settings
#             from google import genai

#             api_key = vision_settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
#             if not api_key:
#                 logger.warning("⚠️ GEMINI_API_KEY not set - Gemini vision will fail")
#                 return

#             # The new SDK uses a Client instance
#             self._client = genai.Client(api_key=api_key)
#             self._model_name = vision_settings.GEMINI_VISION_MODEL

#             logger.info(f"✅ Gemini Client initialized: {self._model_name}")

#         except ImportError:
#             logger.error("❌ Google GenAI SDK not installed. Run: uv add google-genai")
#         except Exception as e:
#             logger.error(f"❌ Gemini initialization failed: {e}")

#     def analyze_image(self, image: Image.Image, prompt: str) -> str:
#         """Analyze dental radiograph using Gemini 2.0/1.5."""
#         from config.visionconfig import vision_settings
#         from google.genai import types

#         self._lazy_load_model()

#         if not self._client:
#             raise RuntimeError("Gemini client not initialized. Check API key.")

#         logger.info(f"🌟 Analyzing with Gemini {self._model_name}...")

#         try:
#             # New SDK supports passing PIL objects directly in the contents list
#             response = self._client.models.generate_content(
#                 model=self._model_name,
#                 contents=[prompt, image],
#                 config=types.GenerateContentConfig(
#                     temperature=vision_settings.VISION_TEMPERATURE,
#                     max_output_tokens=vision_settings.VISION_MAX_TOKENS,
#                 )
#             )

#             if not response.text:
#                 raise RuntimeError("Gemini returned empty response or was blocked by safety filters.")

#             result = response.text
#             logger.info(f"✅ Gemini analysis complete: {len(result)} chars")
#             return result

#         except Exception as e:
#             error_msg = str(e)
#             logger.error(f"❌ Gemini analysis failed: {error_msg}")

#             if "404" in error_msg:
#                 raise RuntimeError(f"Model {self._model_name} not found. Try updating to 'gemini-2.0-flash'.")
#             raise RuntimeError(f"Gemini API error: {error_msg}")

# # Global instance
# gemini_vision_client = GeminiVisionClient()
