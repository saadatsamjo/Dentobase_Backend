# app/cdss_engine/routes.py
"""
CDSS Routes - Updated with tooth_number support
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from app.cdss_engine.fusion_engine import CDSSFusionEngine
from app.cdss_engine.schemas import CDSSRequest, CDSSResponse
from app.database.connection import get_db

router = APIRouter()
cdss_engine = CDSSFusionEngine()

# TODO: Replace with actual token extraction
async def get_current_user_id() -> int:
    """
    Get user ID from JWT token.
    For now, hardcoded to 1.
    
    Later implementation:
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    security = HTTPBearer()
    
    async def get_current_user_id(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> int:
        token = credentials.credentials
        # Decode JWT, extract user_id
        return user_id
    """
    return 1  # Hardcoded for now

@router.post("/provide_final_recommendation", response_model=CDSSResponse)
async def provide_final_recommendation(
    patient_id: int = Form(..., description="Patient database ID"),
    chief_complaint: str = Form(..., description="Main clinical complaint/question"),
    tooth_number: Optional[str] = Form(None, description="Tooth number(s) affected (e.g., '36' or '36,37')"),
    image: Optional[UploadFile] = File(None, description="Dental radiograph (optional)"),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """
    Main CDSS Endpoint - Complete Clinical Decision Support
    
    This endpoint orchestrates:
    1. Fetch patient history from database
    2. Analyze radiograph (if provided) with configured vision model
    3. Retrieve relevant clinical guidelines (RAG)
    4. Generate structured clinical recommendation with configured LLM
    
    **Form Parameters:**
    - **patient_id**: Patient database ID (required)
    - **chief_complaint**: Main complaint or clinical question (required)
    - **tooth_number**: Tooth number(s) affected (optional, e.g., '36' or '36,37')
    - **image**: Dental periapical radiograph (optional, JPEG/PNG)
    
    **Configuration:**
    - Vision Model: Set in config/visionconfig.py (VISION_MODEL_PROVIDER)
    - LLM: Set in config/ragconfig.py (LLM_PROVIDER)
    - Embeddings: Set in config/ragconfig.py (EMBEDDING_PROVIDER)
    
    **Returns:**
    ```json
    {
        "recommendation": {
            "diagnosis": "Primary diagnosis",
            "differential_diagnoses": ["Alt 1", "Alt 2"],
            "recommended_management": "Treatment plan",
            "reference_pages": [100, 101, 102]
        },
        "image_observations": {
            "raw_description": "Full radiograph analysis",
            "pathology_summary": "Key findings",
            "confidence": "high",
            "model_used": "llava"
        },
        "processing_metadata": {
            "vision_provider": "llava",
            "llm_provider": "gpt-4",
            "total_time_seconds": 8.5
        }
    }
    ```
    """
    try:
        # Parse tooth numbers if provided
        tooth_numbers = None
        if tooth_number:
            # Support comma-separated tooth numbers
            tooth_numbers = [t.strip() for t in tooth_number.split(',')]
        
        # Read image if provided
        image_bytes = None
        if image:
            image_bytes = await image.read()
        
        # Execute CDSS pipeline
        result = await cdss_engine.provide_final_recommendation(
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            db=db,
            image_bytes=image_bytes,
            tooth_numbers=tooth_numbers,  # NEW PARAMETER
            user_id=user_id
        )
        
        return result
        
    except ValueError as e:
        # Patient not found or validation error
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Other errors
        raise HTTPException(
            status_code=500, 
            detail=f"CDSS processing error: {str(e)}"
        )

@router.post("/recommendation_json", response_model=CDSSResponse)
async def recommendation_from_json(
    request: CDSSRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """
    CDSS Recommendation from JSON payload (no image upload).
    
    Use when:
    - No radiograph available
    - Testing with JSON payloads
    - Integrating from other systems
    
    **Request Body:**
    ```json
    {
        "patient_id": 123,
        "chief_complaint": "severe toothache lower right",
        "user_id": 1
    }
    ```
    """
    try:
        result = await cdss_engine.provide_final_recommendation(
            patient_id=request.patient_id,
            chief_complaint=request.chief_complaint,
            db=db,
            image_bytes=None,
            tooth_numbers=None,
            user_id=user_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def cdss_health_check():
    """Check CDSS engine configuration and status."""
    from config.visionconfig import vision_settings
    from config.ragconfig import rag_settings
    from app.visionsystem.vision_client import vision_client
    from app.RAGsystem.llm_client import llm_client
    
    return {
        "status": "healthy",
        "configuration": {
            "vision_model": {
                "provider": vision_settings.VISION_MODEL_PROVIDER,
                "info": vision_client.get_model_info()
            },
            "llm": {
                "provider": rag_settings.LLM_PROVIDER,
                "info": llm_client.get_model_info()
            },
            "embeddings": {
                "provider": rag_settings.EMBEDDING_PROVIDER,
                "model": rag_settings.current_embedding_model
            },
            "retriever": {
                "type": rag_settings.RETRIEVER_TYPE,
                "k": rag_settings.RETRIEVAL_K,
                "lambda_multiplier": rag_settings.LAMBDA_MULT if rag_settings.RETRIEVER_TYPE == "mmr" else "N/A",
                "fetch_k": rag_settings.FETCH_K if rag_settings.RETRIEVER_TYPE == "mmr" else "N/A",
                "similarity_threshold": rag_settings.SIMILARITY_THRESHOLD if rag_settings.RETRIEVER_TYPE == "similarity_score_threshold" else "N/A"
            }
        }
    }