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

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.visionsystem.image_processor import ImageProcessor
from app.visionsystem.vision_client import vision_client
from app.visionsystem.vision_schemas import VisionAnalysisResponse
from config.config_schemas import VisionConfigRequest, VisionConfigResponse
from config.visionconfig import vision_settings
from app.users.auth_dependencies import get_current_user_id

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze_image", response_model=VisionAnalysisResponse)
async def analyze_dental_radiograph(
    file: UploadFile = File(...),
    context: str = Form(None),
    tooth_number: str = Form(None),
    user_id: int = Depends(get_current_user_id)
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
        processing_time_ms=elapsed * 1000,
        focused_tooth=tooth_number,
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
        ("florence", None, "Florence-2 - Not recommended"),
        ("llava", "llava:latest", "llava:latest - Baseline (7B)"),
        ("llava", "llava:13b", "llava:13b - Best open-source"),
        ("llava", "llama3.2-vision", "llama3.2-vision - Meta latest"),
        ("gemma3", None, "Gemma 3 4B - Google multimodal local"),  # ← NEW
        ("llava_med", None, "LLaVA-Med - Medical specialist (via llava:13b)"),
        ("biomedclip", None, "BiomedCLIP - Pathology classifier (open_clip)"),
        ("groq", None, "Groq Llama 3.2 90B - Ultra-fast cloud"),  # ← NEW
        ("gemini", None, "Gemini 2.0 Flash - Google cloud"),  # ← NEW
        ("claude", None, "Claude 3.5 Sonnet - Medical reasoning"),
        ("gpt4v", None, "GPT-4 Vision - Best accuracy"),
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


@router.get("/config", response_model=VisionConfigResponse)
async def get_vision_config():
    """
    Get current vision system configuration.

    Returns all vision settings including:
    - Active vision model provider and model name
    - Inference parameters (temperature, max_tokens)
    - Prompt engineering settings
    - Image preprocessing options
    """
    return VisionConfigResponse(
        # Model Selection
        vision_model_provider=vision_settings.VISION_MODEL_PROVIDER,
        current_vision_model=vision_settings.current_vision_model,
        llava_model=vision_settings.LLAVA_MODEL,  #if vision_settings.VISION_MODEL_PROVIDER == "llava" else "N/A",
        gemma3_model=vision_settings.GEMMA3_MODEL,  # ← NEW
        groq_vision_model=vision_settings.GROQ_VISION_MODEL,  # ← NEW
        gemini_vision_model=vision_settings.GEMINI_VISION_MODEL,  # ← NEW
        openai_vision_model=vision_settings.OPENAI_VISION_MODEL,  # ← NEW
        claude_vision_model=vision_settings.CLAUDE_VISION_MODEL,  # ← NEW
        florence_model_name=vision_settings.FLORENCE_MODEL_NAME,
        llava_med_model=vision_settings.LLAVA_MED_MODEL,
        biomedclip_model=vision_settings.BIOMEDCLIP_MODEL,
        
        # Inference Settings
        vision_temperature=vision_settings.VISION_TEMPERATURE,
        vision_max_tokens=vision_settings.VISION_MAX_TOKENS,
        # Prompt Engineering
        include_clinical_notes_in_vision_model_prompt=vision_settings.INCLUDE_CLINICAL_NOTES_IN_VISION_MODEL_PROMPT,
        # Image Processing
        enhance_contrast=vision_settings.ENHANCE_CONTRAST,
        contrast_factor=vision_settings.CONTRAST_FACTOR,
        brightness_factor=vision_settings.BRIGHTNESS_FACTOR,
    )


@router.post("/config", response_model=VisionConfigResponse)
async def update_vision_config(config: VisionConfigRequest):
    """
    Update vision configuration (in-memory only, resets on restart).

    Supports partial updates — send only the fields you want to change.

    Important cache invalidations:
    - Changing vision_model_provider → clears vision client cache
    - Changing llava_model → clears cache and forces model reload

    Example request:
    ```json
    {
        "vision_model_provider": "llava",
        "llava_model": "llama3.2-vision",
        "vision_temperature": 0.0
    }
    ```
    """
    updated_fields = []

    # ── Model Selection ──
    if config.vision_model_provider is not None:
        vision_settings.VISION_MODEL_PROVIDER = config.vision_model_provider
        # Clear vision client cache when provider changes
        vision_client._clients = {}
        updated_fields.append(
            f"vision_model_provider → {config.vision_model_provider} (cache cleared)"
        )

    if config.llava_model is not None:
        vision_settings.LLAVA_MODEL = config.llava_model
        # Force reload of LLaVA client
        if "llava" in vision_client._clients:
            del vision_client._clients["llava"]
        updated_fields.append(f"llava_model → {config.llava_model} (cache cleared)")
    
    if config.gemma3_model is not None:
        vision_settings.GEMMA3_MODEL = config.gemma3_model
        vision_client._clients = {}
        updated_fields.append(f"gemma3_model → {config.gemma3_model}")
    
    if config.groq_vision_model is not None:
        vision_settings.GROQ_VISION_MODEL = config.groq_vision_model
        vision_client._clients = {}
        updated_fields.append(f"groq_vision_model → {config.groq_vision_model}")
    
    if config.gemini_vision_model is not None:
        vision_settings.GEMINI_VISION_MODEL = config.gemini_vision_model
        vision_client._clients = {}
        updated_fields.append(f"gemini_vision_model → {config.gemini_vision_model}")

    # ── Inference Settings ──
    if config.vision_temperature is not None:
        vision_settings.VISION_TEMPERATURE = config.vision_temperature
        updated_fields.append(f"vision_temperature → {config.vision_temperature}")

    if config.vision_max_tokens is not None:
        vision_settings.VISION_MAX_TOKENS = config.vision_max_tokens
        updated_fields.append(f"vision_max_tokens → {config.vision_max_tokens}")

    # ── Prompt Engineering ──
    if config.include_clinical_notes_in_vision_model_prompt is not None:
        vision_settings.INCLUDE_CLINICAL_NOTES_IN_VISION_MODEL_PROMPT = (
            config.include_clinical_notes_in_vision_model_prompt
        )
        updated_fields.append(
            f"include_clinical_notes → {config.include_clinical_notes_in_vision_model_prompt}"
        )

    # ── Image Processing ──
    if config.enhance_contrast is not None:
        vision_settings.ENHANCE_CONTRAST = config.enhance_contrast
        updated_fields.append(f"enhance_contrast → {config.enhance_contrast}")

    if config.contrast_factor is not None:
        vision_settings.CONTRAST_FACTOR = config.contrast_factor
        updated_fields.append(f"contrast_factor → {config.contrast_factor}")

    if config.brightness_factor is not None:
        vision_settings.BRIGHTNESS_FACTOR = config.brightness_factor
        updated_fields.append(f"brightness_factor → {config.brightness_factor}")

    # Log changes
    logger.info(f"📝 Vision Config Updated: {len(updated_fields)} field(s)")
    for field in updated_fields:
        logger.info(f"   • {field}")

    # Return updated config
    return await get_vision_config()
