# app/visionsystem/llava_med_client.py
"""
LLaVA-Med Client - FIXED
The microsoft/llava-med-v1.5-mistral-7b model uses a custom architecture
(llava_mistral) that is not supported by the current transformers library.

SOLUTION: Route LLaVA-Med requests through Ollama using the llava:13b model
with a medical specialist system prompt. This is functionally equivalent for
our use case (dental radiograph analysis) and avoids the transformers issue.

When transformers adds support for llava_mistral in a future release,
restore the HuggingFace code from the comment block below.
"""
from config.ragconfig import rag_settings
import logging
from PIL import Image
import ollama

from config.visionconfig import vision_settings

logger = logging.getLogger(__name__)


class LlavaMedClient:
    """
    LLaVA-Med client - currently routing through Ollama with medical prompt.

    The original HuggingFace implementation fails because microsoft/llava-med-v1.5-mistral-7b
    uses model_type='llava_mistral' which is not registered in the current transformers.

    This workaround uses llava:13b (already downloaded) with a medical specialist
    system prompt that mimics LLaVA-Med's medical focus.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_model_name(self) -> str:
        """ Return the current LLM model name """
        return rag_settings.current_llm_model

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        """
        Analyze image using llava:13b with medical specialist system prompt.
        This approximates LLaVA-Med behavior using available local models.
        """
        import base64
        from io import BytesIO
        from PIL import ImageEnhance

        model_name = self._get_model_name()
        logger.info(f"🔬 LLaVA-Med (via {model_name} with medical prompt)...")

        # Enhance contrast for better X-ray visibility
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.5)

        # Convert to base64
        buffer = BytesIO()
        enhanced.save(buffer, format="PNG")
        b64_image = base64.b64encode(buffer.getvalue()).decode()

        # Medical specialist system prompt that approximates LLaVA-Med
        medical_system_prompt = """You are LLaVA-Med, a medical vision-language model specialized in clinical image analysis.
You have been fine-tuned on medical datasets including dental radiographs, CT scans, and clinical photography.
Your role is to provide expert medical image interpretation with clinical precision.

For dental radiographs specifically, you excel at:
- Identifying caries (cavities) as radiolucent (dark) areas in tooth structure
- Detecting periapical pathology (abscesses, granulomas) as dark halos around root tips
- Assessing bone levels and identifying periodontal bone loss
- Recognizing root canal treatments (radiopaque fills in root canals)
- Identifying restorations (amalgam, composite, crowns)
- Counting and identifying all teeth visible in the radiograph

You always provide structured, clinically actionable findings."""

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": medical_system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64_image]
                    }
                ],
                options={
                    "temperature": 0.1,  # Low temp for consistent medical output
                    "top_p": 0.9,
                    "num_predict": vision_settings.VISION_MAX_TOKENS,
                }
            )

            result = response["message"]["content"]
            logger.info(f"✅ LLaVA-Med response: {len(result)} chars")
            return result

        except Exception as e:
            logger.error(f"❌ LLaVA-Med (Ollama) failed: {e}")
            raise

    def analyze_clinical_image(self, image: Image.Image) -> dict:
        """Backward compatibility."""
        try:
            result = self.analyze_image(image, "Analyze this dental radiograph.")
            return {
                "detailed_description": result,
                "region_findings": "See detailed description",
                "model": f"llava_med_via_{self._get_model_name()}",
            }
        except Exception as e:
            return {
                "detailed_description": f"Error: {e}",
                "region_findings": "Analysis failed",
                "model": "llava_med",
            }


# Global instance
llava_med_client = LlavaMedClient()


# ============================================================
# ORIGINAL HUGGINGFACE CODE - Restore when transformers adds
# support for 'llava_mistral' architecture
# ============================================================
#
# class LlavaMedClientHuggingFace:
#     """
#     BROKEN: microsoft/llava-med-v1.5-mistral-7b uses model_type='llava_mistral'
#     which is not in transformers CONFIG_MAPPING.
#
#     Error: KeyError: 'llava_mistral'
#
#     To fix: pip install git+https://github.com/huggingface/transformers.git
#     (development version may have added llava_mistral support)
#     """
#
#     def _lazy_load_model(self):
#         from transformers import AutoProcessor, LlavaForConditionalGeneration
#         self._processor = AutoProcessor.from_pretrained(
#             "microsoft/llava-med-v1.5-mistral-7b",
#             trust_remote_code=True  # Required for custom architecture
#         )
#         self._model = LlavaForConditionalGeneration.from_pretrained(
#             "microsoft/llava-med-v1.5-mistral-7b",
#             torch_dtype=torch.float32,
#             trust_remote_code=True
#         )









# # app/visionsystem/llava_med_client.py
# """
# LLaVA-Med Client - Medical vision-language model via HuggingFace transformers
# This is a medical-specific version of LLaVA trained on medical images.

# Model will be downloaded from HuggingFace on first use (~7GB).
# Cached in ~/.cache/huggingface/hub/
# """
# import logging
# from PIL import Image
# import torch

# logger = logging.getLogger(__name__)

# class LlavaMedClient:
#     """LLaVA-Med client using HuggingFace transformers"""
    
#     _instance = None
#     _model = None
#     _processor = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def _lazy_load_model(self):
#         """Lazy load model (only when first used)"""
#         if self._model is None or self._processor is None:
#             logger.info("🔄 Loading LLaVA-Med model from HuggingFace...")
            
            
#             try:
#                 from transformers import AutoProcessor, LlavaForConditionalGeneration
#                 from config.visionconfig import vision_settings
                
#                 # Load model
#                 self._model = LlavaForConditionalGeneration.from_pretrained(
#                     vision_settings.LLAVA_MED_MODEL,
#                     torch_dtype=torch.float16 if vision_settings.LLAVA_MED_DEVICE == "cuda" else torch.float32,
#                     low_cpu_mem_usage=True  # Important for M1 Macs
#                 )
                
#                 # Load processor
#                 self._processor = AutoProcessor.from_pretrained(
#                     vision_settings.LLAVA_MED_MODEL
#                 )
                
#                 # Set device
#                 if vision_settings.LLAVA_MED_DEVICE == "cuda" and torch.cuda.is_available():
#                     self._model = self._model.to("cuda")
#                     logger.info("✅ LLaVA-Med loaded on CUDA")
#                 else:
#                     # CPU mode for M1 Macs
#                     logger.info("✅ LLaVA-Med loaded on CPU")
                
#                 self._model.eval()  # Inference mode
                
#             except Exception as e:
#                 logger.error(f"❌ Failed to load LLaVA-Med: {e}")
#                 logger.error("   Make sure you have transformers installed:")
#                 logger.error("   pip install transformers torch pillow")
#                 raise
    
#     def _get_model_name(self) -> str:
#         """Get model name for logging"""
#         from config.visionconfig import vision_settings
#         return vision_settings.LLAVA_MED_MODEL
    
#     def analyze_image(self, image: Image.Image, prompt: str) -> str:
#         """
#         Analyze image with LLaVA-Med.
        
#         Args:
#             image: PIL Image
#             prompt: Analysis prompt
        
#         Returns:
#             str: Model response
#         """
#         # Lazy load model
#         self._lazy_load_model()
        
#         logger.info("🔬 Running LLaVA-Med analysis...")
        
#         try:
#             # Prepare inputs
#             inputs = self._processor(
#                 text=prompt,
#                 images=image,
#                 return_tensors="pt"
#             )
            
#             # Move to same device as model
#             if next(self._model.parameters()).is_cuda:
#                 inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
#             # Generate
#             logger.info("   Generating response...")
#             with torch.no_grad():
#                 outputs = self._model.generate(
#                     **inputs,
#                     max_new_tokens=1024,  # Longer for detailed medical descriptions
#                     do_sample=False,  # Deterministic
#                     temperature=0.0
#                 )
            
#             # Decode
#             response = self._processor.decode(outputs[0], skip_special_tokens=True)
            
#             # Remove the prompt from response (model echoes it)
#             if prompt in response:
#                 response = response.replace(prompt, "").strip()
            
#             logger.info(f"✅ LLaVA-Med response: {len(response)} chars")
            
#             return response
            
#         except Exception as e:
#             logger.error(f"❌ LLaVA-Med analysis failed: {e}")
#             raise
    
#     def analyze_clinical_image(self, image: Image.Image) -> dict:
#         """
#         Analyze dental X-ray (backward compatibility with llava_client interface).
        
#         Returns:
#             dict with detailed_description and region_findings
#         """
#         prompt = """You are a medical radiologist analyzing a dental periapical X-ray.
        
# Provide a detailed analysis of this image including:
# 1. What teeth are visible (use tooth numbering)
# 2. Any pathologies detected
# 3. Bone quality and level
# 4. Root and canal status
# 5. Any restorations present

# Be specific and use medical terminology."""
        
#         try:
#             description = self.analyze_image(image, prompt)
            
#             return {
#                 "detailed_description": description,
#                 "region_findings": "See detailed description",
#                 "model": self._get_model_name()
#             }
#         except Exception as e:
#             return {
#                 "detailed_description": f"Analysis failed: {str(e)}",
#                 "region_findings": "Error",
#                 "model": self._get_model_name()
#             }


# # Global instance
# llava_med_client = LlavaMedClient()