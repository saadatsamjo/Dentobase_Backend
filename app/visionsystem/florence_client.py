# app/visionsystem/florence_client.py
import os

os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"

import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from config.visionconfig import vision_settings

logger = logging.getLogger(__name__)


class FlorenceClient:
    """Production-grade Florence-2 client for clinical image analysis."""

    _instance = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self):
        """Load Florence-2 with clinical settings."""
        if self._model is not None:
            return self._processor, self._model

        logger.info(f"Loading Florence-2: {vision_settings.FLORENCE_MODEL_NAME}")
        device = self._get_device()

        self._processor = AutoProcessor.from_pretrained(
            vision_settings.FLORENCE_MODEL_NAME, trust_remote_code=True
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            vision_settings.FLORENCE_MODEL_NAME,
            trust_remote_code=True,
            attn_implementation="eager",
            dtype=torch.float32,
            device_map=None,
        )

        self._model = self._model.to(device)
        self._model.eval()

        logger.info(f"✅Florence-2 ready on {device}")
        return self._processor, self._model

    def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """Clinical image analysis."""
        processor, model = self.load_model()
        device = next(model.parameters()).device

        task_prompt = prompt or "<MORE_DETAILED_CAPTION>"

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process inputs
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Create generation config to override model defaults
        gen_config = GenerationConfig(
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=False,  # Disable KV cache to avoid beam search bug
        )

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                generation_config=gen_config,
            )

        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse
        try:
            parsed = processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(image.width, image.height)
            )

            if isinstance(parsed, dict) and task_prompt in parsed:
                result = parsed[task_prompt]
                logger.info(f"Analysis complete: {len(str(result))} chars")
                return str(result)
            else:
                return str(parsed)

        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            # Fallback: clean up raw text
            cleaned = generated_text.replace(task_prompt, "").strip()
            cleaned = cleaned.replace("</s>", "").replace("<s>", "").strip()
            return cleaned

    def analyze_clinical_image(self, image: Image.Image) -> dict:
        """Comprehensive clinical analysis with multiple prompts."""
        return {
            "detailed_description": self.analyze_image(image, "<MORE_DETAILED_CAPTION>"),
            "region_findings": self.analyze_image(image, "<DENSE_REGION_CAPTION>"),
            "model": vision_settings.FLORENCE_MODEL_NAME,
        }


florence_client = FlorenceClient()
