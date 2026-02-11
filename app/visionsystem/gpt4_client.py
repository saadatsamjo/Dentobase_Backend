# app/visionsystem/gpt4_client.py
"""
GPT-4 Vision Client - Optimized for Dental Radiographs
FIXED: Using gpt-4o (current model, not deprecated)
"""
import os
import base64
import logging
from io import BytesIO

from PIL import Image
from openai import OpenAI

from config.visionconfig import vision_settings

logger = logging.getLogger(__name__)

# Dental radiograph prompt for GPT-4V
DENTAL_XRAY_PROMPT = """Analyze this DENTAL PERIAPICAL RADIOGRAPH (X-ray image).

            YOUR TASK: Identify all pathologies and clinical abnormalities present in this dental X-ray.

            IMAGE INFORMATION:
            - Type: Periapical radiograph (shows tooth root and surrounding bone)
            - Purpose: Diagnostic evaluation for dental pathology

            SYSTEMATIC ANALYSIS REQUIRED:

            1. IMAGE VERIFICATION:
            - Confirm this is a periapical dental X-ray
            - Assess diagnostic quality

            2. ANATOMICAL STRUCTURES:
            - Identify visible tooth/teeth (number if possible)
            - Crown, root(s), pulp chamber, root canals
            - Alveolar bone, periodontal ligament, lamina dura

            3. PATHOLOGY DETECTION (PRIMARY FOCUS):

            A. DENTAL CARIES:
                - Radiolucent (dark) areas in tooth crown
                - Location: mesial, distal, occlusal, buccal, lingual
                - Depth: enamel, dentin, approaching pulp
            
            B. PERIAPICAL PATHOLOGY:
                - Radiolucent area at root apex = infection/abscess/cyst
                - Size estimation (mm)
                - Characteristics: well-defined, diffuse
                - PDL widening
            
            C. BONE LOSS:
                - Horizontal bone loss: reduced bone height
                - Vertical bone loss: angular defects
                - Distance from CEJ to alveolar crest
            
            D. PULPAL STATUS:
                - Pulp stones or calcifications
                - Chamber obliteration
                - Previous root canal treatment evident
            
            E. ROOT ABNORMALITIES:
                - Resorption (internal/external)
                - Fractures
                - Morphological variations
            
            F. RESTORATIONS:
                - Radiopaque (bright) materials = fillings, crowns
                - Quality: overhangs, recurrent decay
                - Material type if identifiable

            4. RADIOGRAPHIC TERMINOLOGY:
            - RADIOLUCENT (dark) = Lower density: caries, infection, bone loss
            - RADIOPAQUE (bright) = Higher density: enamel, restorations, bone

            5. CLINICAL SIGNIFICANCE:
            - Primary diagnosis
            - Severity level: Mild / Moderate / Severe
            - Clinical urgency: Routine / Prompt / Urgent
            - Treatment implications

            RESPONSE FORMAT:
            - Be SPECIFIC with tooth numbers and surfaces
            - Use PROPER DENTAL TERMINOLOGY
            - Provide MEASUREMENTS when possible
            - State clearly if NO PATHOLOGY DETECTED

            Provide your detailed clinical analysis:"""


class GPT4VisionClient:
    """GPT-4 Vision client for dental radiograph analysis."""

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _get_model_name(self):
        """Get GPT model - FIXED to use gpt-4o instead of deprecated gpt-4-vision-preview."""
        return "gpt-4o"  # Current model with vision capabilities

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """Analyze dental radiograph using GPT-4o."""
        client = self._get_client()
        b64_image = self._encode_image(image)
        model_name = self._get_model_name()

        analysis_prompt = prompt or DENTAL_XRAY_PROMPT

        logger.info(f"Analyzing dental radiograph with {model_name}...")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "high",  # High detail for clinical accuracy
                                },
                            },
                        ],
                    }
                ],
                max_tokens=vision_settings.VISION_MAX_TOKENS,
                temperature=vision_settings.VISION_TEMPERATURE,
            )

            result = response.choices[0].message.content
            logger.info(f"GPT-4o analysis complete: {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"GPT-4o analysis failed: {e}")
            raise

    def analyze_clinical_image(self, image: Image.Image) -> dict:
        """Comprehensive dental radiograph analysis."""
        # Main detailed analysis
        detailed = self.analyze_image(image, DENTAL_XRAY_PROMPT)

        # Optional pathology summary
        pathology = None
        if vision_settings.DUAL_PROMPT_ANALYSIS:
            try:
                pathology_prompt = """Provide a concise pathology summary from this dental X-ray:

                        PATHOLOGY CHECKLIST:
                        1. Caries: Yes/No (if yes: location, depth)
                        2. Periapical lesion: Yes/No (if yes: size, tooth affected)
                        3. Bone loss: Yes/No (if yes: type, severity)
                        4. Restorations: Yes/No (if yes: type, condition)
                        5. Other findings: List any other abnormalities

                        Format as structured bullet points."""
                pathology = self.analyze_image(image, pathology_prompt)
            except Exception as e:
                logger.error(f"Pathology summary failed: {e}")

        return {
            "detailed_description": detailed,
            "region_findings": pathology or "See detailed description",
            "model": self._get_model_name(),
        }


# Global instance
gpt4v_client = GPT4VisionClient()