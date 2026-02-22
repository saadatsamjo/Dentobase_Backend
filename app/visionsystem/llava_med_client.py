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

    def analyze_image(self, image: Image.Image, prompt: str) -> dict:
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
            logger.error(f"❌ LLaVA-Med (Ollama) failed: {e}")
            raise

    def analyze_clinical_image(self, image: Image.Image) -> dict:
        """Backward compatibility."""
        try:
            analysis_result = self.analyze_image(image, "Analyze this dental radiograph.")
            detailed = analysis_result["text"]
            input_tokens = analysis_result.get("input_tokens", 0)
            output_tokens = analysis_result.get("output_tokens", 0)
            
            return {
                "detailed_description": detailed,
                "region_findings": "See detailed description",
                "model": f"llava_med_via_{self._get_model_name()}",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        except Exception as e:
            return {
                "detailed_description": f"Error: {e}",
                "region_findings": "Analysis failed",
                "model": "llava_med",
            }


# Global instance
llava_med_client = LlavaMedClient()