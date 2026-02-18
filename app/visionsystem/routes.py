# app/visionsystem/routes.py
"""
Vision System Routes - FIXED v2
Fixes:
1. llava:7b -> llava:latest (actual installed model name from ollama list)
2. BiomedCLIP properly loads via open_clip (open-clip-torch is installed)
3. LLaVA-Med routes through Ollama with medical system prompt
4. GPT-4V and Claude gracefully report billing errors without crashing
"""
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.visionsystem.image_processor import ImageProcessor
from app.visionsystem.vision_client import vision_client
from app.visionsystem.vision_schemas import VisionAnalysisResponse
from config.visionconfig import vision_settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze_image", response_model=VisionAnalysisResponse)
async def analyze_dental_radiograph(
    file: UploadFile = File(...),
    context: str = Form(None),
    tooth_number: str = Form(None),
):
    """Analyze a single dental radiograph using the configured vision model."""
    logger.info(f"\n{'='*70}")
    logger.info(f"SINGLE MODEL VISION ANALYSIS")
    logger.info(f"{'='*70}")
    logger.info(f"🔍 Vision Model Provider: {vision_settings.VISION_MODEL_PROVIDER}")
    logger.info(f"🔍 Vision Model: {vision_settings.current_vision_model}")
    if context:
        logger.info(f"📝 Chief Complaint: {context}")
    if tooth_number:
        logger.info(f"🦷 Focus Tooth: #{tooth_number}")

    content = await file.read()
    image = ImageProcessor.preprocess_image(content)

    start_time = time.time()
    result = vision_client.analyze_dental_radiograph(
        image, context=context, tooth_number=tooth_number
    )
    elapsed = time.time() - start_time
    logger.info(f"✅ Analysis complete in {elapsed:.2f}s")

    return VisionAnalysisResponse(
        detailed_description=result["detailed_description"],
        pathology_summary=result.get("pathology_summary", ""),
        model_used=result.get("model", vision_settings.current_vision_model),
        image_quality_score=result.get("image_quality_score", 0.5),
        diagnostic_confidence=result.get("confidence_score", 0.5),
        structured_findings=result.get("structured_findings"),
    )


@router.post("/test_vision_models")
async def test_all_vision_models(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None),
    tooth_number: Optional[str] = Form(None),
):
    """
    TEST ENDPOINT: Compare ALL available vision models on the same image.

    Models tested:
    - llava:13b       (Ollama - best open-source)
    - llama3.2-vision (Ollama - Meta latest)
    - llava:latest    (Ollama - baseline, installed as llava:latest not llava:7b)
    - llava_med       (Ollama + medical system prompt, routes via llava:13b)
    - biomedclip      (open_clip - real BiomedCLIP classification)
    - florence        (HuggingFace Florence-2-base)
    - gpt4v           (OpenAI - skipped if no credits)
    - claude          (Anthropic - skipped if no credits)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"MULTI-MODEL VISION COMPARISON TEST")
    logger.info(f"{'='*70}")
    if context:
        logger.info(f"📝 Test Context: {context}")
    if tooth_number:
        logger.info(f"🦷 Test Tooth: #{tooth_number}")

    content = await file.read()
    image = ImageProcessor.preprocess_image(content)

    # (provider_type, ollama_model_override_or_None, label)
    # NOTE: llava:7b is NOT installed - it's llava:latest (4.7 GB, same model)
    models_to_test = [
        ("llava",     "llava:13b",          "llava:13b - Recommended open-source"),
        ("llava",     "llama3.2-vision",    "llama3.2-vision - Latest from Meta"),
        ("llava",     "llava:latest",       "llava:latest - Baseline (7B)"),
        ("llava_med", None,                 "LLaVA-Med - Medical specialist (via llava:13b)"),
        ("biomedclip", None,                "BiomedCLIP - Pathology classifier (open_clip)"),
        ("florence",  None,                 "Florence-2 - Not recommended"),
        ("gpt4v",     None,                 "GPT-4 Vision - Best accuracy"),
        ("claude",    None,                 "Claude 3.5 Sonnet - Medical reasoning"),
    ]

    results = {}
    original_provider = vision_settings.VISION_MODEL_PROVIDER
    original_llava_model = vision_settings.LLAVA_MODEL

    for provider, ollama_model, description in models_to_test:
        logger.info(f"\n{'─'*70}")
        logger.info(f"🔄 Testing: {provider} ({description})")
        logger.info(f"{'─'*70}")

        model_label = ollama_model if ollama_model else provider
        result_key = f"{provider}_{model_label.replace(':', '_').replace('.', '_')}"

        try:
            # Switch provider
            vision_settings.VISION_MODEL_PROVIDER = provider
            vision_client._clients = {}  # Force reload of client

            # Also switch the Ollama model if specified
            if ollama_model:
                vision_settings.LLAVA_MODEL = ollama_model
                logger.info(f"   🔧 Ollama model set to: {ollama_model}")

            start = time.time()
            result = vision_client.analyze_dental_radiograph(
                image, context=context, tooth_number=tooth_number
            )
            elapsed = (time.time() - start) * 1000

            if result.get("error"):
                logger.error(f"❌ {provider} returned error: {result['error']}")
                results[result_key] = {
                    "success": False,
                    "description": description,
                    "error": result["error"],
                }
                continue

            structured = result.get("structured_findings")
            results[result_key] = {
                "success": True,
                "description": description,
                "model_used": model_label,
                "structured_findings": structured,
                "teeth_visible": structured.get("teeth_visible") if structured else None,
                "primary_finding": structured.get("primary_finding") if structured else None,
                "pathology_summary": result.get("pathology_summary", ""),
                "image_quality": result.get("image_quality_score", 0.0),
                "confidence": result.get("confidence_score", 0.0),
                "time_ms": round(elapsed, 2),
            }

            logger.info(f"✅ Success - {elapsed:.0f}ms")
            if structured:
                logger.info(f"   Teeth: {structured.get('teeth_visible', [])}")
                logger.info(f"   Finding: {str(structured.get('primary_finding', ''))[:70]}...")

        except Exception as e:
            err_str = str(e)
            # Classify error type for cleaner reporting
            if "insufficient_quota" in err_str or "credit balance" in err_str:
                err_msg = "API billing: no credits. Add credits to use this model."
            elif "404" in err_str and "not found" in err_str:
                err_msg = f"Model not installed in Ollama: {model_label}"
            else:
                err_msg = err_str[:200]

            logger.error(f"❌ {provider} failed: {err_msg}")
            results[result_key] = {
                "success": False,
                "description": description,
                "error": err_msg,
            }
        finally:
            # Always restore after each test
            vision_settings.VISION_MODEL_PROVIDER = original_provider
            vision_settings.LLAVA_MODEL = original_llava_model
            vision_client._clients = {}

    # Final restore
    vision_settings.VISION_MODEL_PROVIDER = original_provider
    vision_settings.LLAVA_MODEL = original_llava_model

    successful = sum(1 for r in results.values() if r.get("success"))
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"✅ {successful}/{len(results)} models succeeded")

    return {
        "test_context": {
            "context": context,
            "tooth_number": tooth_number,
            "models_tested": len(results),
            "successful": successful,
        },
        "results": results,
        "recommendation": {
            "best_open_source": "llava:13b or llama3.2-vision",
            "fastest_baseline": "llava:latest (same as llava:7b)",
            "medical_specialist": "llava_med (llava:13b + medical prompt)",
            "pathology_classifier": "biomedclip (BiomedCLIP via open_clip)",
            "best_accuracy": "gpt4v or claude (requires API credits)",
        },
    }


@router.get("/config")
async def get_vision_config():
    """Return current vision model configuration."""
    return {
        "provider": vision_settings.VISION_MODEL_PROVIDER,
        "llava_model": vision_settings.LLAVA_MODEL,
        "max_tokens": vision_settings.VISION_MAX_TOKENS,
        "temperature": vision_settings.VISION_TEMPERATURE,
        "enhance_contrast": vision_settings.ENHANCE_CONTRAST,
    }