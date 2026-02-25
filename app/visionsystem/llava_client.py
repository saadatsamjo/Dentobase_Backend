# app/visionsystem/llava_client.py
"""
LLaVA Vision Client - FIXED with proper instance export
"""
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import ollama
import logging
from config.visionconfig import vision_settings

logger = logging.getLogger(__name__)

# Dental radiograph analysis prompt (unchanged from your original)
DENTAL_XRAY_PROMPT = """You are analyzing a DENTAL PERIAPICAL RADIOGRAPH (X-ray).
PURPOSE: Identify pathologies, anomalies, and clinical findings in this dental X-ray image.

CRITICAL CONTEXT:
- This is a MEDICAL IMAGE, specifically a dental X-ray
- You are looking at tooth/teeth and surrounding bone structures
- Your job is to detect ABNORMALITIES and PATHOLOGIES

SYSTEMATIC ANALYSIS REQUIRED:
1. IMAGE TYPE CONFIRMATION
2. ANATOMICAL STRUCTURES
3. PATHOLOGY DETECTION (MOST IMPORTANT)
4. RADIOGRAPHIC TERMINOLOGY
5. CLINICAL SIGNIFICANCE

OUTPUT REQUIREMENTS:
✅ Be SPECIFIC
✅ Use DENTAL TERMINOLOGY
✅ State SEVERITY levels
✅ If NO pathology found, state clearly

Begin your analysis:"""

PATHOLOGY_CHECKLIST_PROMPT = """FOCUSED PATHOLOGY DETECTION:
Answer YES or NO for each category, with specific details if YES:
1. CARIES (Cavities)
2. PERIAPICAL LESION/ABSCESS
3. BONE LOSS
4. ROOT CANAL TREATMENT
5. RESTORATIONS (Fillings/Crowns)
6. OTHER ABNORMALITIES

FORMAT: Structured bullet points, clinical terminology."""


class LlavaClient:
    """LLaVA client optimized for dental radiograph analysis."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_model_name(self):
        """Get LLaVA model name from settings."""
        return vision_settings.LLAVA_MODEL
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 with optional X-ray enhancement."""
        if vision_settings.ENHANCE_CONTRAST:
            # Enhance contrast for better X-ray visibility
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(vision_settings.CONTRAST_FACTOR)
            
            # Enhance brightness
            brightness_enhancer = ImageEnhance.Brightness(image)
            image = brightness_enhancer.enhance(vision_settings.BRIGHTNESS_FACTOR)
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def analyze_image(self, image: Image.Image, prompt: str = None) -> dict:
        """
        Analyze dental image using LLaVA.
        
        Args:
            image: PIL Image (dental radiograph)
            prompt: Custom prompt or None for default dental analysis
        
        Returns:
            Dict with 'text', 'input_tokens', 'output_tokens'
        """
        model_name = self._get_model_name()
        
        # Use dental-specific prompt if none provided
        analysis_prompt = prompt or DENTAL_XRAY_PROMPT
        
        # Convert image to base64
        b64_image = self._encode_image(image)
        
        logger.info(f"Analyzing dental radiograph with {model_name}...")
        
        try:
            # Call Ollama with clinical settings
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert dental radiologist with 20+ years of experience in periapical radiograph interpretation. 
You specialize in detecting dental pathologies including caries, periapical lesions, bone loss, and root pathology.
Your analyses are precise, use proper dental terminology, and focus on clinically significant findings."""
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt,
                        "images": [b64_image]
                    }
                ],
                options={
                    "temperature": vision_settings.VISION_TEMPERATURE,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": vision_settings.VISION_MAX_TOKENS,
                }
            )
            
            result = response["message"]["content"]
            logger.info(f"LLaVA analysis complete: {len(result)} characters")
            
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
            logger.error(f"LLaVA analysis failed: {e}")
            raise
    
    def analyze_clinical_image(self, image: Image.Image) -> dict:
        """
        Comprehensive dental radiograph analysis with dual prompts.
        
        Returns:
            dict with detailed_description and region_findings (pathology summary)
        """
        input_tokens = 0
        output_tokens = 0
        
        # Main detailed analysis
        try:
            analysis_result = self.analyze_image(image, DENTAL_XRAY_PROMPT)
            detailed = analysis_result["text"]
            input_tokens += analysis_result.get("input_tokens", 0)
            output_tokens += analysis_result.get("output_tokens", 0)
        except Exception as e:
            logger.error(f"Detailed analysis failed: {e}")
            detailed = f"Analysis failed: {str(e)}"
        
        # Pathology-focused checklist (if dual prompt enabled)
        pathology = None
        if vision_settings.DUAL_PROMPT_ANALYSIS:
            try:
                pathology_result = self.analyze_image(image, PATHOLOGY_CHECKLIST_PROMPT)
                pathology = pathology_result["text"]
                input_tokens += pathology_result.get("input_tokens", 0)
                output_tokens += pathology_result.get("output_tokens", 0)
            except Exception as e:
                logger.error(f"Pathology checklist failed: {e}")
                pathology = "Pathology assessment unavailable"
        
        return {
            "detailed_description": detailed,
            "region_findings": pathology or "See detailed description",
            "model": self._get_model_name(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }


# CRITICAL: Global instance export
llava_client = LlavaClient()