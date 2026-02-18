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
    
    def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Analyze dental image using LLaVA.
        
        Args:
            image: PIL Image (dental radiograph)
            prompt: Custom prompt or None for default dental analysis
        
        Returns:
            Clinical description string
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
            return result
            
        except Exception as e:
            logger.error(f"LLaVA analysis failed: {e}")
            raise
    
    def analyze_clinical_image(self, image: Image.Image) -> dict:
        """
        Comprehensive dental radiograph analysis with dual prompts.
        
        Returns:
            dict with detailed_description and region_findings (pathology summary)
        """
        # Main detailed analysis
        try:
            detailed = self.analyze_image(image, DENTAL_XRAY_PROMPT)
        except Exception as e:
            logger.error(f"Detailed analysis failed: {e}")
            detailed = f"Analysis failed: {str(e)}"
        
        # Pathology-focused checklist (if dual prompt enabled)
        pathology = None
        if vision_settings.DUAL_PROMPT_ANALYSIS:
            try:
                pathology = self.analyze_image(image, PATHOLOGY_CHECKLIST_PROMPT)
            except Exception as e:
                logger.error(f"Pathology checklist failed: {e}")
                pathology = "Pathology assessment unavailable"
        
        return {
            "detailed_description": detailed,
            "region_findings": pathology or "See detailed description",
            "model": self._get_model_name()
        }


# CRITICAL: Global instance export
llava_client = LlavaClient()





# # app/visionsystem/llava_client.py
# """
# LLaVA Vision Client - Optimized for Dental Radiographs
# """
# import base64
# from io import BytesIO
# from PIL import Image, ImageEnhance
# import ollama
# import logging
# from config.visionconfig import vision_settings

# logger = logging.getLogger(__name__)

# # ============================================================================
# # DENTAL RADIOGRAPH ANALYSIS PROMPT
# # ============================================================================
# DENTAL_XRAY_PROMPT = """You are analyzing a DENTAL PERIAPICAL RADIOGRAPH (X-ray).

# PURPOSE: Identify pathologies, anomalies, and clinical findings in this dental X-ray image.

# CRITICAL CONTEXT:
# - This is a MEDICAL IMAGE, specifically a dental X-ray
# - You are looking at tooth/teeth and surrounding bone structures
# - Your job is to detect ABNORMALITIES and PATHOLOGIES

# SYSTEMATIC ANALYSIS REQUIRED:

# 1. IMAGE TYPE CONFIRMATION:
#    - Confirm this is a periapical radiograph (shows tooth root and surrounding bone)
#    - Note image quality and diagnostic value

# 2. ANATOMICAL STRUCTURES:
#    - Which tooth/teeth are visible? (Use tooth numbering if identifiable)
#    - Crown, root(s), pulp chamber, root canal(s)
#    - Surrounding alveolar bone
#    - Periodontal ligament space
#    - Lamina dura (white line around root)

# 3. PATHOLOGY DETECTION (MOST IMPORTANT):
   
#    a) DENTAL CARIES:
#       - Look for RADIOLUCENT (dark) areas in the crown
#       - Location: occlusal, mesial, distal, buccal, lingual
#       - Depth: enamel only, into dentin, approaching pulp
   
#    b) PERIAPICAL PATHOLOGY:
#       - RADIOLUCENT area at root apex = INFECTION/ABSCESS
#       - Widened PDL space
#       - Loss of lamina dura
#       - Size and characteristics of any periapical radiolucency
   
#    c) BONE LOSS:
#       - Horizontal bone loss (bone level lower than normal)
#       - Vertical bone loss (angular defects)
#       - Measure from CEJ to alveolar crest
   
#    d) PULPAL PATHOLOGY:
#       - Pulp stones/calcifications
#       - Pulp chamber obliteration
#       - Evidence of previous root canal (radiopaque filling material)
   
#    e) ROOT PATHOLOGY:
#       - Root resorption (internal or external)
#       - Root fractures
#       - Abnormal root morphology
   
#    f) EXISTING DENTAL WORK:
#       - RADIOPAQUE (bright white) = metal fillings, crowns
#       - Note quality: overhangs, recurrent decay, margins

# 4. RADIOGRAPHIC TERMINOLOGY:
#    - RADIOLUCENT = DARK = low density (decay, infection, air, soft tissue)
#    - RADIOPAQUE = BRIGHT = high density (enamel, bone, metal, dentin)

# 5. CLINICAL SIGNIFICANCE:
#    - Severity: Mild / Moderate / Severe
#    - Urgency: Routine / Prompt / Urgent / Emergency
#    - Primary diagnosis based on findings

# OUTPUT REQUIREMENTS:
# ✅ Be SPECIFIC: "Tooth #19 shows mesial caries" not "there is decay"
# ✅ Use DENTAL TERMINOLOGY
# ✅ State SEVERITY levels
# ✅ If NO pathology found, state clearly: "No significant pathology detected"
# ❌ Do NOT describe this as flowers, plants, or non-medical objects
# ❌ Do NOT use vague terms like "appears to show"

# Begin your analysis:"""

# PATHOLOGY_CHECKLIST_PROMPT = """FOCUSED PATHOLOGY DETECTION:

# Answer YES or NO for each category, with specific details if YES:

# 1. CARIES (Cavities):
#    Present? If YES: Which tooth, which surface, depth estimation

# 2. PERIAPICAL LESION/ABSCESS:
#    Present? If YES: Which tooth, approximate size, characteristics

# 3. BONE LOSS:
#    Present? If YES: Type (horizontal/vertical), severity (mild/moderate/severe)

# 4. ROOT CANAL TREATMENT:
#    Present? If YES: Which tooth, quality of obturation

# 5. RESTORATIONS (Fillings/Crowns):
#    Present? If YES: Type, condition, any secondary caries

# 6. OTHER ABNORMALITIES:
#    Present? If YES: Specify (fractures, resorption, calcifications, etc.)

# FORMAT: Structured bullet points, clinical terminology."""

# class LlavaClient:
#     """LLaVA client optimized for dental radiograph analysis."""
    
#     _instance = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def _get_model_name(self):
#         """Get LLaVA model name from settings."""
#         return vision_settings.LLAVA_MODEL
    
#     def _encode_image(self, image: Image.Image) -> str:
#         """Convert PIL Image to base64 with optional X-ray enhancement."""
#         if vision_settings.ENHANCE_CONTRAST:
#             # Enhance contrast for better X-ray visibility
#             enhancer = ImageEnhance.Contrast(image)
#             image = enhancer.enhance(vision_settings.CONTRAST_FACTOR)
            
#             # Enhance brightness
#             brightness_enhancer = ImageEnhance.Brightness(image)
#             image = brightness_enhancer.enhance(vision_settings.BRIGHTNESS_FACTOR)
        
#         buffer = BytesIO()
#         image.save(buffer, format="PNG")
#         return base64.b64encode(buffer.getvalue()).decode()
    
#     def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
#         """
#         Analyze dental image using LLaVA.
        
#         Args:
#             image: PIL Image (dental radiograph)
#             prompt: Custom prompt or None for default dental analysis
        
#         Returns:
#             Clinical description string
#         """
#         model_name = self._get_model_name()
        
#         # Use dental-specific prompt if none provided
#         analysis_prompt = prompt or DENTAL_XRAY_PROMPT
        
#         # Convert image to base64
#         b64_image = self._encode_image(image)
        
#         logger.info(f"Analyzing dental radiograph with {model_name}...")
        
#         try:
#             # Call Ollama with clinical settings
#             response = ollama.chat(
#                 model=model_name,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": """You are an expert dental radiologist with 20+ years of experience in periapical radiograph interpretation. 
# You specialize in detecting dental pathologies including caries, periapical lesions, bone loss, and root pathology.
# Your analyses are precise, use proper dental terminology, and focus on clinically significant findings."""
#                     },
#                     {
#                         "role": "user",
#                         "content": analysis_prompt,
#                         "images": [b64_image]
#                     }
#                 ],
#                 options={
#                     "temperature": vision_settings.VISION_TEMPERATURE,
#                     "top_p": 0.9,
#                     "top_k": 40,
#                     "num_predict": vision_settings.VISION_MAX_TOKENS,
#                 }
#             )
            
#             result = response["message"]["content"]
#             logger.info(f"LLaVA analysis complete: {len(result)} characters")
#             return result
            
#         except Exception as e:
#             logger.error(f"LLaVA analysis failed: {e}")
#             raise
    
#     def analyze_clinical_image(self, image: Image.Image) -> dict:
#         """
#         Comprehensive dental radiograph analysis with dual prompts.
        
#         Returns:
#             dict with detailed_description and region_findings (pathology summary)
#         """
#         # Main detailed analysis
#         try:
#             detailed = self.analyze_image(image, DENTAL_XRAY_PROMPT)
#         except Exception as e:
#             logger.error(f"Detailed analysis failed: {e}")
#             detailed = f"Analysis failed: {str(e)}"
        
#         # Pathology-focused checklist (if dual prompt enabled)
#         pathology = None
#         if vision_settings.DUAL_PROMPT_ANALYSIS:
#             try:
#                 pathology = self.analyze_image(image, PATHOLOGY_CHECKLIST_PROMPT)
#             except Exception as e:
#                 logger.error(f"Pathology checklist failed: {e}")
#                 pathology = "Pathology assessment unavailable"
        
#         return {
#             "detailed_description": detailed,
#             "region_findings": pathology or "See detailed description",
#             "model": self._get_model_name()
#         }

# # Global instance
# llava_client = LlavaClient()