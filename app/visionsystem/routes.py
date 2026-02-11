# app/visionsystem/routes.py
"""
Vision System Routes - FIXED
Properly returns pathology summary from dual-prompt analysis
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import time
import logging

from app.visionsystem.vision_client import vision_client
from app.visionsystem.image_processor import ImageProcessor
from app.cdss_engine.schemas import VisionAnalysisResponse
from config.visionconfig import vision_settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze_image", response_model=VisionAnalysisResponse)
async def analyze_dental_radiograph(
    file: UploadFile = File(..., description="Dental radiograph image"),
    custom_prompt: Optional[str] = Form(None, description="Optional custom analysis prompt")
):
    """
    Analyze Dental Radiograph with Dual-Prompt Analysis
    
    Returns both:
    - detailed_description: Full clinical analysis
    - pathology_summary: Structured checklist of findings
    
    Vision Model: Configured in config/visionconfig.py
    """
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"VISION ENDPOINT - RADIOGRAPH ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"📸 Image: {file.filename}")
        logger.info(f"🔍 Vision Model: {vision_settings.VISION_MODEL_PROVIDER}")
        logger.info(f"   Dual-prompt: {vision_settings.DUAL_PROMPT_ANALYSIS}")
        
        # Validate image
        content = await file.read()
        ImageProcessor.validate_image(file.content_type, len(content))
        logger.info(f"✓ Image validated: {len(content)} bytes")
        
        # Preprocess
        image = ImageProcessor.preprocess_image(content)
        logger.info(f"✓ Image preprocessed: {image.size}")
        
        # Analyze with configured vision model
        if custom_prompt:
            logger.info(f"⚠️  Using custom prompt (pathology summary disabled)")
            # Custom prompt analysis
            description = vision_client.analyze_image(image, custom_prompt)
            result = {
                "detailed_description": description,
                "pathology_summary": "Custom analysis - see detailed description",
                "model": vision_client.get_model_info()["model"],
                "confidence": "medium"
            }
        else:
            logger.info(f"✓ Using standard dental radiograph analysis")
            # Standard dental radiograph analysis with dual prompts
            result = vision_client.analyze_dental_radiograph(image)
            
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
        
        logger.info(f"✓ Analysis complete")
        logger.info(f"   Model: {result.get('model', vision_settings.VISION_MODEL_PROVIDER)}")
        logger.info(f"   Description: {len(result['detailed_description'])} chars")
        logger.info(f"   Pathology summary: {len(result.get('pathology_summary', ''))} chars")
        logger.info(f"   Processing time: {processing_time:.2f}ms")
        logger.info(f"   Confidence: {result.get('confidence', 'medium')}")
        logger.info(f"{'='*60}\n")
        
        return VisionAnalysisResponse(
            detailed_description=result["detailed_description"],
            pathology_summary=result.get("pathology_summary", "See detailed description"),
            model_used=result.get("model", vision_settings.VISION_MODEL_PROVIDER),
            processing_time_ms=round(processing_time, 2),
            confidence=result.get("confidence", "medium")
        )
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Vision analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Vision analysis failed: {str(e)}"
        )

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
            "supported_formats": vision_settings.SUPPORTED_FORMATS
        },
        "prompts": {
            "detailed_analysis": "Comprehensive radiograph analysis with anatomical structures and pathology",
            "pathology_checklist": "Structured Yes/No checklist for caries, bone loss, lesions, etc." if vision_settings.DUAL_PROMPT_ANALYSIS else "Disabled"
        }
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
    
    models_to_test = ["llava", "gpt4v", "claude", "florence"]
    
    for model_name in models_to_test:
        logger.info(f"Testing {model_name}...")
        
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
                "detailed_description": result["detailed_description"][:200] + "..." if len(result["detailed_description"]) > 200 else result["detailed_description"],
                "pathology_summary": result.get("pathology_summary", "")[:100] + "..." if result.get("pathology_summary") and len(result.get("pathology_summary", "")) > 100 else result.get("pathology_summary", ""),
                "time_ms": round(elapsed, 2),
                "model_used": result.get("model", model_name)
            }
            
            logger.info(f"   ✓ {model_name}: {elapsed:.2f}ms")
            
            # Restore original
            vision_settings.VISION_MODEL_PROVIDER = original_provider
            vision_client._clients = {}
            
        except Exception as e:
            logger.error(f"   ❌ {model_name}: {str(e)}")
            results[model_name] = {
                "success": False,
                "error": str(e)
            }
    
    logger.info(f"{'='*60}\n")
    
    return {
        "test_results": results,
        "recommendation": "gpt4v or claude for production (highest accuracy), llava:13b for development (good balance)",
        "note": "Florence-2 not recommended for medical imaging"
    }
    
    
    
    
# # app/visionsystem/routes.py
# """
# Vision System Routes - Refactored
# Uses unified vision client that routes to configured model
# """
# from fastapi import APIRouter, UploadFile, File, HTTPException, Form
# from typing import Optional
# import time
# import logging

# from app.visionsystem.vision_client import vision_client
# from app.visionsystem.image_processor import ImageProcessor
# from app.cdss_engine.schemas import VisionAnalysisResponse
# from config.visionconfig import vision_settings

# router = APIRouter()
# logger = logging.getLogger(__name__)

# @router.post("/analyze_image", response_model=VisionAnalysisResponse)
# async def analyze_dental_radiograph(
#     file: UploadFile = File(..., description="Dental radiograph image"),
#     custom_prompt: Optional[str] = Form(None, description="Optional custom analysis prompt")
# ):
#     """
#     Analyze Dental Radiograph
    
#     Standalone vision analysis endpoint - analyzes radiograph WITHOUT patient context.
#     For complete CDSS with patient history, use /api/cdss/provide_final_recommendation instead.
    
#     **Vision Model Selection:**
#     Configure in `config/visionconfig.py`:
#     - VISION_MODEL_PROVIDER: "llava", "gpt4v", "claude", or "florence"
#     - LLAVA_MODEL: "llava", "llava:13b", or "llava:34b" (if using LLaVA)
    
#     **Input:**
#     - **file**: Dental periapical radiograph (JPEG, PNG, WebP)
#     - **custom_prompt**: Optional custom instructions (default: dental pathology detection)
    
#     **Output:**
#     ```json
#     {
#         "detailed_description": "Full clinical analysis of radiograph",
#         "pathology_summary": "Caries: present, Bone loss: mild, etc.",
#         "model_used": "llava:13b",
#         "processing_time_ms": 3420.5,
#         "confidence": "high"
#     }
#     ```
#     """
#     start_time = time.time()
    
#     try:
#         logger.info(f"Analyzing radiograph: {file.filename}")
#         logger.info(f"Using vision model: {vision_settings.VISION_MODEL_PROVIDER}")
        
#         # Validate image
#         content = await file.read()
#         ImageProcessor.validate_image(file.content_type, len(content))
        
#         # Preprocess
#         image = ImageProcessor.preprocess_image(content)
        
#         # Analyze with configured vision model
#         if custom_prompt:
#             # Custom prompt analysis
#             description = vision_client.analyze_image(image, custom_prompt)
#             result = {
#                 "detailed_description": description,
#                 "pathology_summary": "Custom analysis - see detailed description",
#                 "model": vision_client.get_model_info()["model"],
#                 "confidence": "medium"
#             }
#         else:
#             # Standard dental radiograph analysis
#             result = vision_client.analyze_dental_radiograph(image)
#             result["confidence"] = "high" if "no pathology" not in result["detailed_description"].lower() else "medium"
        
#         processing_time = (time.time() - start_time) * 1000
        
#         return VisionAnalysisResponse(
#             detailed_description=result["detailed_description"],
#             pathology_summary=result.get("pathology_summary", "See detailed description"),
#             model_used=result.get("model", vision_settings.VISION_MODEL_PROVIDER),
#             processing_time_ms=round(processing_time, 2),
#             confidence=result.get("confidence", "medium")
#         )
        
#     except ValueError as e:
#         logger.error(f"Validation error: {e}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Vision analysis failed: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Vision analysis failed: {str(e)}"
#         )

# @router.get("/config")
# async def get_vision_config():
#     """
#     Get current vision system configuration.
    
#     Returns which vision model is currently active and its settings.
#     """
#     return {
#         "vision_provider": vision_settings.VISION_MODEL_PROVIDER,
#         "model_details": vision_client.get_model_info(),
#         "settings": {
#             "max_image_size": vision_settings.MAX_IMAGE_SIZE,
#             "enhance_contrast": vision_settings.ENHANCE_CONTRAST,
#             "dual_prompt_analysis": vision_settings.DUAL_PROMPT_ANALYSIS,
#             "supported_formats": vision_settings.SUPPORTED_FORMATS
#         }
#     }

# @router.post("/test_models")
# async def test_all_vision_models(file: UploadFile = File(...)):
#     """
#     Test image with all available vision models (for comparison).
    
#     **Warning**: This will make API calls to GPT-4V and Claude if configured.
#     Use only for testing/comparison purposes.
#     """
#     results = {}
#     content = await file.read()
#     image = ImageProcessor.preprocess_image(content)
    
#     # Test each model
#     models_to_test = ["llava", "gpt4v", "claude", "florence"]
    
#     for model_name in models_to_test:
#         try:
#             # Temporarily switch to this model
#             original_provider = vision_settings.VISION_MODEL_PROVIDER
#             vision_settings.VISION_MODEL_PROVIDER = model_name
            
#             start = time.time()
#             result = vision_client.analyze_dental_radiograph(image)
#             elapsed = (time.time() - start) * 1000
            
#             results[model_name] = {
#                 "success": True,
#                 "detailed_description": result["detailed_description"][:200] + "...",
#                 "pathology_summary": result.get("pathology_summary", "")[:100] + "...",
#                 "time_ms": round(elapsed, 2)
#             }
            
#             # Restore original
#             vision_settings.VISION_MODEL_PROVIDER = original_provider
            
#         except Exception as e:
#             results[model_name] = {
#                 "success": False,
#                 "error": str(e)
#             }
    
#     return {
#         "test_results": results,
#         "recommendation": "gpt4v or claude for production, llava for development"
#     }