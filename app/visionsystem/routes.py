# app/visionsystem/routes.py (PARTIAL - Updated analyze_image endpoint)
"""
Vision Analysis Routes - Updated with clinical context support
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import time
import logging

from app.visionsystem.vision_client import vision_client
from app.visionsystem.image_processor import ImageProcessor
from app.cdss_engine.schemas import VisionAnalysisResponse
from app.system_models.clinical_note_model.clinical_note_model import ClinicalNote
from app.database.connection import get_db
from config.visionconfig import vision_settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze_image", response_model=VisionAnalysisResponse)
async def analyze_dental_radiograph(
    file: UploadFile = File(..., description="Dental radiograph image"),
    patient_id: Optional[int] = Form(None, description="Patient ID for clinical context"),
    chief_complaint: Optional[str] = Form(None, description="Chief complaint or clinical question"),
    tooth_number: Optional[str] = Form(None, description="Specific tooth number(s) to focus on"),
    custom_prompt: Optional[str] = Form(None, description="Optional custom analysis prompt"),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze Dental Radiograph with Clinical Context
    
    This endpoint now supports clinical context integration:
    - Accepts patient_id to retrieve recent clinical notes
    - Accepts chief_complaint for focused analysis
    - Accepts tooth_number for tooth-specific examination
    
    Returns both:
    - detailed_description: Full clinical analysis
    - pathology_summary: Structured checklist of findings
    
    Vision Model: Configured in config/visionconfig.py
    """
    start_time = time.time()

    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"VISION ENDPOINT - RADIOGRAPH ANALYSIS")
        logger.info(f"{'='*70}")
        logger.info(f"📸 Image: {file.filename}")
        logger.info(f"🔍 Vision Model: {vision_settings.VISION_MODEL_PROVIDER}")
        logger.info(f"   Dual-prompt: {vision_settings.DUAL_PROMPT_ANALYSIS}")
        
        if patient_id:
            logger.info(f"👤 Patient ID: {patient_id}")
        if chief_complaint:
            logger.info(f"📋 Chief Complaint: {chief_complaint}")
        if tooth_number:
            logger.info(f"🦷 Tooth Number(s): {tooth_number}")

        # Validate image
        content = await file.read()
        ImageProcessor.validate_image(file.content_type, len(content))
        logger.info(f"✅ Image validated: {len(content)} bytes")

        # Preprocess
        image = ImageProcessor.preprocess_image(content)
        logger.info(f"✅ Image preprocessed: {image.size}")

        # Build clinical context
        clinical_context = None
        clinical_notes_text = None
        
        if chief_complaint or tooth_number or patient_id:
            context_parts = []
            
            if chief_complaint:
                context_parts.append(f"Chief complaint: {chief_complaint}")
            
            if tooth_number:
                tooth_list = tooth_number.replace(',', ', ')
                context_parts.append(f"Focus on tooth/teeth: {tooth_list}")
            
            clinical_context = ". ".join(context_parts)
            
            # Fetch clinical notes if patient_id provided
            if patient_id and db:
                logger.info(f"📝 Fetching recent clinical notes for patient {patient_id}...")
                try:
                    notes_query = (
                        select(ClinicalNote)
                        .join(ClinicalNote.encounter)
                        .where(ClinicalNote.encounter.has(patient_id=patient_id))
                        .order_by(desc(ClinicalNote.created_at))
                        .limit(2)  # Get 2 most recent notes
                    )
                    
                    notes_result = await db.execute(notes_query)
                    clinical_notes = notes_result.scalars().all()
                    
                    if clinical_notes:
                        logger.info(f"✅ Retrieved {len(clinical_notes)} clinical note(s)")
                        notes_parts = []
                        for note in clinical_notes:
                            notes_parts.append(f"[{note.note_type}]: {note.content}")
                        clinical_notes_text = " | ".join(notes_parts)
                        logger.info(f"   Notes summary: {clinical_notes_text[:100]}...")
                    else:
                        logger.info(f"   No clinical notes found")
                except Exception as e:
                    logger.warning(f"⚠️  Could not fetch clinical notes: {e}")

        # Analyze with configured vision model
        if custom_prompt:
            logger.info(f"⚠️  Using custom prompt (pathology summary disabled)")
            # Custom prompt analysis
            description = vision_client.analyze_image(image, custom_prompt)
            result = {
                "detailed_description": description,
                "pathology_summary": "Custom analysis - see detailed description",
                "model": vision_client.get_model_info()["model"],
                "confidence": "medium",
            }
        else:
            logger.info(f"✅ Using context-aware dental radiograph analysis")
            if clinical_context:
                logger.info(f"   Context: {clinical_context}")
            
            # Standard dental radiograph analysis with clinical context
            result = vision_client.analyze_dental_radiograph(
                image, 
                context=clinical_context,
                clinical_notes=clinical_notes_text
            )

            # Determine confidence based on content
            description_lower = result["detailed_description"].lower()
            if "no pathology" in description_lower or "no significant" in description_lower:
                confidence = "medium"
            elif "severe" in description_lower or "urgent" in description_lower:
                confidence = "high"
            else:
                confidence = "high"

            result["confidence"] = confidence

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"✅ Analysis complete")
        logger.info(f"   Model: {result.get('model', vision_settings.VISION_MODEL_PROVIDER)}")
        logger.info(f"   Description: {len(result['detailed_description'])} chars")
        logger.info(f"   Pathology summary: {len(result.get('pathology_summary', ''))} chars")
        logger.info(f"   Processing time: {processing_time:.2f}ms")
        logger.info(f"   Confidence: {result.get('confidence', 'medium')}")
        
        # Enhanced logging - show pathology summary
        logger.info(f"\n{'─'*70}")
        logger.info(f"📊 IMAGE ANALYSIS RESULTS:")
        logger.info(f"{'─'*70}")
        logger.info(f"Model Used: {result.get('model', vision_settings.VISION_MODEL_PROVIDER)}")
        logger.info(f"Confidence: {result.get('confidence', 'medium')}")
        logger.info(f"\n📋 Pathology Summary:")
        logger.info(f"{result.get('pathology_summary', 'No summary available')}")
        logger.info(f"{'─'*70}")
        logger.info(f"{'='*70}\n")

        return VisionAnalysisResponse(
            detailed_description=result["detailed_description"],
            pathology_summary=result.get("pathology_summary", "See detailed description"),
            model_used=result.get("model", vision_settings.VISION_MODEL_PROVIDER),
            processing_time_ms=round(processing_time, 2),
            confidence=result.get("confidence", "medium"),
        )

    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Vision analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vision analysis failed: {str(e)}")



@router.get("/config")
async def get_vision_config():
    """Get current vision system configuration."""
    return {
        "vision_provider": vision_settings.VISION_MODEL_PROVIDER,
        "model_details": vision_client.get_model_info(),
        "settings": {
            "max_image_size": vision_settings.MAX_IMAGE_SIZE,
            "enhance_contrast": vision_settings.ENHANCE_CONTRAST,
            "dual_prompt_analysis": vision_settings.DUAL_PROMPT_ANALYSIS,
            "contrast_factor": vision_settings.CONTRAST_FACTOR,
            "brightness_factor": vision_settings.BRIGHTNESS_FACTOR,
            "supported_formats": vision_settings.SUPPORTED_FORMATS,
        },
        "prompts": {
            "detailed_analysis": "Comprehensive radiograph analysis with anatomical structures and pathology",
            "pathology_checklist": (
                "Structured Yes/No checklist for caries, bone loss, lesions, etc."
                if vision_settings.DUAL_PROMPT_ANALYSIS
                else "Disabled"
            ),
        },
    }


@router.post("/test_models")
async def test_all_vision_models(file: UploadFile = File(...)):
    """
    Test image with all available vision models for comparison.

    Warning: Makes API calls to GPT-4 and Claude if configured.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING ALL VISION MODELS")
    logger.info(f"{'='*60}")

    results = {}
    content = await file.read()
    image = ImageProcessor.preprocess_image(content)

    models_to_test = ["florence", "llava", "gpt4v", "claude"]

    for model_name in models_to_test:
        logger.info(f"Testing {model_name}...")
        logger.info("......................................................................")

        try:
            # Temporarily switch to this model
            original_provider = vision_settings.VISION_MODEL_PROVIDER
            vision_settings.VISION_MODEL_PROVIDER = model_name

            # Force reload of client
            vision_client._clients = {}
            
            start = time.time()
            result = vision_client.analyze_dental_radiograph(image)
            elapsed = (time.time() - start) * 1000

            results[model_name] = {
                "success": True,
                "detailed_description": (
                    result["detailed_description"][:200] + "..."
                    if len(result["detailed_description"]) > 200
                    else result["detailed_description"]
                ),
                "pathology_summary": (
                    result.get("pathology_summary", "")[:100] + "..."
                    if result.get("pathology_summary")
                    and len(result.get("pathology_summary", "")) > 100
                    else result.get("pathology_summary", "")
                ),
                "time_ms": round(elapsed, 2),
                "model_used": result.get("model", model_name),
            }

            logger.info(f"✅ {model_name}: Time elapsed: {elapsed:.2f}ms")
            

            # Restore original
            vision_settings.VISION_MODEL_PROVIDER = original_provider
            vision_client._clients = {}

        except Exception as e:
            logger.info("......................................................................")
            logger.error(f"❌ {model_name}: {str(e)}")
            results[model_name] = {"success": False, "error": str(e)}
            logger.info("......................................................................")

    logger.info(f"{'='*60}\n")

    return {
        "test_results": results,
        # "recommendation": "gpt4v or claude for production (highest accuracy), llava:13b for development (good balance)",
        # "note": "Florence-2 not recommended for medical imaging"
    }
