# app/RAGsystem/routes.py - COMPLETE UPDATE
"""
RAG Routes with Schema Normalization
Ensures consistent response structure for frontend
"""
import logging
import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.RAGsystem.chains import ClinicalRAGChain
from app.RAGsystem.embeddings import embedding_provider
from app.RAGsystem.retriever import RetrieverFactory
from app.RAGsystem.schemas import DocumentUploadResponse, QuestionPayload, RAGResponse
from app.RAGsystem.response_normalizer import normalize_rag_response, validate_response
from app.users.auth_dependencies import get_current_user_id
from config.config_schemas import RAGConfigRequest, RAGConfigResponse
from config.ragconfig import rag_settings
from scripts.ingest_documents import ingest_pdf

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/answer_question", response_model=RAGResponse)
async def answer_question_endpoint(payload: QuestionPayload):
    """
    RAG endpoint with guaranteed consistent schema.
    
    Features:
    - Normalizes LLM response variations to single standard format
    - Ensures reference pages are always present
    - Validates response before returning
    
    Returns standardized JSON suitable for frontend consumption.
    """
    if not payload.question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        # Get RAG chain and invoke
        retriever = RetrieverFactory().get_retriever()
        chain = ClinicalRAGChain(retriever)
        raw_answer = chain.invoke(payload.question)
        
        logger.info(f"📝 Raw LLM response: {len(raw_answer)} chars")
        
        # Parse JSON
        try:
            parsed_data = json.loads(raw_answer)
            logger.info("✅ Parsed LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.error(f"❌ LLM returned invalid JSON: {e}")
            raise HTTPException(
                status_code=500,
                detail="LLM did not return valid JSON. Please try again or contact support."
            )
        
        # NORMALIZE - Convert to standard schema
        normalized_data = normalize_rag_response(parsed_data)
        logger.info(f"🔄 Normalized response - found {len(normalized_data.get('reference_pages', []))} reference pages")
        
        # VALIDATE - Ensure all required fields present
        validate_response(normalized_data)
        logger.info("✅ Response validated successfully")
        
        return RAGResponse(
            answer=normalized_data,
            retrieval_strategy=rag_settings.RETRIEVER_TYPE
        )
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"❌ RAG endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing question: {str(e)}"
        )


@router.get("/config", response_model=RAGConfigResponse)
async def get_rag_config():
    """Get current RAG system configuration."""
    return RAGConfigResponse(
        # LLM
        llm_provider=rag_settings.LLM_PROVIDER,
        current_llm_model=rag_settings.current_llm_model,
        ollama_llm_model=rag_settings.OLLAMA_LLM_MODEL,
        openai_llm_model=rag_settings.OPENAI_LLM_MODEL,
        claude_llm_model=rag_settings.CLAUDE_LLM_MODEL,
        groq_llm_model=rag_settings.GROQ_LLM_MODEL,
        gemini_llm_model=rag_settings.GEMINI_LLM_MODEL,
        # Embedding
        embedding_provider=rag_settings.EMBEDDING_PROVIDER,
        current_embedding_model=rag_settings.current_embedding_model,
        ollama_embedding_model=rag_settings.OLLAMA_EMBEDDING_MODEL,
        hf_embedding_model=rag_settings.HF_EMBEDDING_MODEL,
        # Retrieval
        retriever_type=rag_settings.RETRIEVER_TYPE,
        retrieval_k=rag_settings.RETRIEVAL_K,
        fetch_k=rag_settings.FETCH_K,
        lambda_mult=rag_settings.LAMBDA_MULT,
        similarity_threshold=rag_settings.SIMILARITY_THRESHOLD,
        # Generation
        llm_temperature=rag_settings.LLM_TEMPERATURE,
        max_tokens=rag_settings.MAX_TOKENS,
        # Processing
        chunk_size=rag_settings.CHUNK_SIZE,
        chunk_overlap=rag_settings.CHUNK_OVERLAP,
        pdf_path=rag_settings.PDF_PATH,
        persist_dir=rag_settings.PERSIST_DIR,
        device=rag_settings.effective_device,
    )


@router.post("/config", response_model=RAGConfigResponse)
async def update_rag_config(config: RAGConfigRequest):
    """Update RAG configuration with cache invalidation."""
    updated_fields = []
    
    # LLM Provider and Models
    if config.llm_provider is not None:
        rag_settings.LLM_PROVIDER = config.llm_provider
        updated_fields.append(f"llm_provider → {config.llm_provider}")
    
    if config.ollama_llm_model is not None:
        rag_settings.OLLAMA_LLM_MODEL = config.ollama_llm_model
        updated_fields.append(f"ollama_llm_model → {config.ollama_llm_model}")
    
    if config.openai_llm_model is not None:
        rag_settings.OPENAI_LLM_MODEL = config.openai_llm_model
        updated_fields.append(f"openai_llm_model → {config.openai_llm_model}")
    
    if config.claude_llm_model is not None:
        rag_settings.CLAUDE_LLM_MODEL = config.claude_llm_model
        updated_fields.append(f"claude_llm_model → {config.claude_llm_model}")
    
    if config.groq_llm_model is not None:
        rag_settings.GROQ_LLM_MODEL = config.groq_llm_model
        updated_fields.append(f"groq_llm_model → {config.groq_llm_model}")
    
    if config.gemini_llm_model is not None:
        rag_settings.GEMINI_LLM_MODEL = config.gemini_llm_model
        updated_fields.append(f"gemini_llm_model → {config.gemini_llm_model}")
    
    # Embedding Provider
    if config.embedding_provider is not None:
        old_provider = rag_settings.EMBEDDING_PROVIDER
        rag_settings.EMBEDDING_PROVIDER = config.embedding_provider
        updated_fields.append(f"embedding_provider → {config.embedding_provider}")
        
        # Cache invalidation
        if old_provider != config.embedding_provider:
            embedding_provider.refresh()
            RetrieverFactory()._retriever = None
            logger.info("🔄 Cleared embedding cache")
    
    if config.ollama_embedding_model is not None:
        rag_settings.OLLAMA_EMBEDDING_MODEL = config.ollama_embedding_model
        updated_fields.append(f"ollama_embedding_model → {config.ollama_embedding_model}")
        embedding_provider.refresh()
    
    if config.hf_embedding_model is not None:
        rag_settings.HF_EMBEDDING_MODEL = config.hf_embedding_model
        updated_fields.append(f"hf_embedding_model → {config.hf_embedding_model}")
        embedding_provider.refresh()
    
    # Retrieval Settings
    if config.retriever_type is not None:
        old_type = rag_settings.RETRIEVER_TYPE
        rag_settings.RETRIEVER_TYPE = config.retriever_type
        updated_fields.append(f"retriever_type → {config.retriever_type}")
        
        if old_type != config.retriever_type:
            RetrieverFactory()._retriever = None
            logger.info("🔄 Cleared retriever cache")
    
    if config.retrieval_k is not None:
        rag_settings.RETRIEVAL_K = config.retrieval_k
        updated_fields.append(f"retrieval_k → {config.retrieval_k}")
    
    if config.fetch_k is not None:
        rag_settings.FETCH_K = config.fetch_k
        updated_fields.append(f"fetch_k → {config.fetch_k}")
    
    if config.lambda_mult is not None:
        rag_settings.LAMBDA_MULT = config.lambda_mult
        updated_fields.append(f"lambda_mult → {config.lambda_mult}")
    
    if config.similarity_threshold is not None:
        rag_settings.SIMILARITY_THRESHOLD = config.similarity_threshold
        updated_fields.append(f"similarity_threshold → {config.similarity_threshold}")
    
    # Generation Settings
    if config.llm_temperature is not None:
        rag_settings.LLM_TEMPERATURE = config.llm_temperature
        updated_fields.append(f"llm_temperature → {config.llm_temperature}")
    
    if config.max_tokens is not None:
        rag_settings.MAX_TOKENS = config.max_tokens
        updated_fields.append(f"max_tokens → {config.max_tokens}")
    
    # Document Processing
    if config.chunk_size is not None:
        rag_settings.CHUNK_SIZE = config.chunk_size
        updated_fields.append(f"chunk_size → {config.chunk_size}")
    
    if config.chunk_overlap is not None:
        rag_settings.CHUNK_OVERLAP = config.chunk_overlap
        updated_fields.append(f"chunk_overlap → {config.chunk_overlap}")
    
    logger.info(f"📝 RAG Config Updated: {len(updated_fields)} field(s)")
    for field in updated_fields:
        logger.info(f"   • {field}")
    
    return await get_rag_config()


@router.post("/upload_document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id)
):
    """Upload and ingest PDF into vector store."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    temp_path = Path(f"/tmp/{file.filename}")
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest into vector store
        await run_in_threadpool(ingest_pdf, str(temp_path))
        
        # Clear retriever cache
        RetrieverFactory()._retriever = None
        
        logger.info(f"✅ Uploaded and ingested: {file.filename}")
        
        return DocumentUploadResponse(
            message="Document uploaded and ingested successfully",
            filename=file.filename,
            success=True
        )
    
    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path.exists():
            temp_path.unlink()
            
            
# # app/RAGsystem/routes.py
# import logging
# import os
# import shutil
# from pathlib import Path

# from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
# from pydantic import BaseModel
# from starlette.concurrency import run_in_threadpool

# from app.RAGsystem.chains import ClinicalRAGChain, answer_question
# from app.RAGsystem.embeddings import embedding_provider
# from app.RAGsystem.retriever import RetrieverFactory, get_retriever
# from app.RAGsystem.schemas import DocumentUploadResponse, QuestionPayload, RAGResponse
# from app.users.auth_dependencies import get_current_user_id
# from config.config_schemas import RAGConfigRequest, RAGConfigResponse
# from config.ragconfig import rag_settings
# from scripts.ingest_documents import ingest_pdf
# import json  # Add at top of file

# logger = logging.getLogger(__name__)

# router = APIRouter()



# @router.post("/answer_question", response_model=RAGResponse)
# async def answer_question_endpoint(payload: QuestionPayload):
#     """RAG endpoint with proper JSON parsing"""
#     if not payload.question:
#         raise HTTPException(status_code=400, detail="Question is required")

#     retriever = RetrieverFactory().get_retriever()
#     chain = ClinicalRAGChain(retriever)
#     raw_answer = chain.invoke(payload.question)
    
#     # Parse JSON string to dict (THE FIX)
#     try:
#         parsed_answer = json.loads(raw_answer)
#         logger.info("✅ Parsed RAG response as JSON")
#     except json.JSONDecodeError:
#         logger.warning("⚠️  RAG response not valid JSON, returning as-is")
#         parsed_answer = {"raw_text": raw_answer}
    
#     return RAGResponse(
#         answer=parsed_answer,  # Now returns dict, not string!
#         retrieval_strategy=rag_settings.RETRIEVER_TYPE
#     )


# @router.get("/config", response_model=RAGConfigResponse)
# async def get_rag_config():
#     """
#     Get current RAG system configuration.

#     Returns all RAG settings including:
#     - LLM provider and models
#     - Embedding provider and models
#     - Retrieval strategy and parameters
#     - Generation settings
#     - Document processing settings
#     """
#     return RAGConfigResponse(
#         # LLM
#         llm_provider=rag_settings.LLM_PROVIDER,
#         current_llm_model=rag_settings.current_llm_model,
#         ollama_llm_model=rag_settings.OLLAMA_LLM_MODEL,
#         openai_llm_model=rag_settings.OPENAI_LLM_MODEL,
#         claude_llm_model=rag_settings.CLAUDE_LLM_MODEL,
#         groq_llm_model=rag_settings.GROQ_LLM_MODEL,  # ← NEW
#         gemini_llm_model=rag_settings.GEMINI_LLM_MODEL,  # ← NEW
#         # Embedding
#         embedding_provider=rag_settings.EMBEDDING_PROVIDER,
#         current_embedding_model=rag_settings.current_embedding_model,
#         ollama_embedding_model=rag_settings.OLLAMA_EMBEDDING_MODEL,
#         hf_embedding_model=rag_settings.HF_EMBEDDING_MODEL,
#         # Retrieval
#         retriever_type=rag_settings.RETRIEVER_TYPE,
#         retrieval_k=rag_settings.RETRIEVAL_K,
#         fetch_k=rag_settings.FETCH_K,
#         lambda_mult=rag_settings.LAMBDA_MULT,
#         similarity_threshold=rag_settings.SIMILARITY_THRESHOLD,
#         # Generation
#         llm_temperature=rag_settings.LLM_TEMPERATURE,
#         max_tokens=rag_settings.MAX_TOKENS,
#         # Processing
#         chunk_size=rag_settings.CHUNK_SIZE,
#         chunk_overlap=rag_settings.CHUNK_OVERLAP,
#         pdf_path=rag_settings.PDF_PATH,
#         persist_dir=rag_settings.PERSIST_DIR,
#         device=rag_settings.effective_device,
#     )


# @router.post("/config", response_model=RAGConfigResponse)
# async def update_rag_config(config: RAGConfigRequest):
#     """
#     Update RAG configuration (in-memory only, resets on restart).

#     Supports partial updates — send only the fields you want to change.

#     Important cache invalidations:
#     - Changing retriever_type → clears retriever cache
#     - Changing embedding_provider → refreshes embeddings (may need re-ingestion)

#     Example request:
#     ```json
#     {
#         "retriever_type": "mmr",
#         "retrieval_k": 10,
#         "lambda_mult": 0.7
#     }
#     ```
#     """
#     updated_fields = []

#     # ── LLM Provider and Models ──
#     if config.llm_provider is not None:
#         rag_settings.LLM_PROVIDER = config.llm_provider
#         updated_fields.append(f"llm_provider → {config.llm_provider}")

#     if config.ollama_llm_model is not None:
#         rag_settings.OLLAMA_LLM_MODEL = config.ollama_llm_model
#         updated_fields.append(f"ollama_llm_model → {config.ollama_llm_model}")

#     if config.openai_llm_model is not None:
#         rag_settings.OPENAI_LLM_MODEL = config.openai_llm_model
#         updated_fields.append(f"openai_llm_model → {config.openai_llm_model}")

#     if config.claude_llm_model is not None:
#         rag_settings.CLAUDE_LLM_MODEL = config.claude_llm_model
#         updated_fields.append(f"claude_llm_model → {config.claude_llm_model}")
        
#     if config.groq_llm_model is not None:
#         rag_settings.GROQ_LLM_MODEL = config.groq_llm_model
#         updated_fields.append(f"groq_llm_model → {config.groq_llm_model}")

#     if config.gemini_llm_model is not None:
#         rag_settings.GEMINI_LLM_MODEL = config.gemini_llm_model
#         updated_fields.append(f"gemini_llm_model → {config.gemini_llm_model}")

#     # ── Embedding Provider ──
#     if config.embedding_provider is not None:
#         rag_settings.EMBEDDING_PROVIDER = config.embedding_provider
#         embedding_provider.refresh()  # Reload embeddings
#         updated_fields.append(
#             f"embedding_provider → {config.embedding_provider} (embeddings refreshed)"
#         )

#     if config.ollama_embedding_model is not None:
#         rag_settings.OLLAMA_EMBEDDING_MODEL = config.ollama_embedding_model
#         if rag_settings.EMBEDDING_PROVIDER == "ollama":
#             embedding_provider.refresh()
#         updated_fields.append(f"ollama_embedding_model → {config.ollama_embedding_model}")

#     if config.hf_embedding_model is not None:
#         rag_settings.HF_EMBEDDING_MODEL = config.hf_embedding_model
#         if rag_settings.EMBEDDING_PROVIDER == "huggingface":
#             embedding_provider.refresh()
#         updated_fields.append(f"hf_embedding_model → {config.hf_embedding_model}")

#     # ── Retrieval Strategy ──
#     if config.retriever_type is not None:
#         rag_settings.RETRIEVER_TYPE = config.retriever_type
#         # Clear retriever cache when strategy changes
#         retriever_factory = RetrieverFactory()
#         retriever_factory._retriever = None
#         updated_fields.append(f"retriever_type → {config.retriever_type} (cache cleared)")

#     if config.retrieval_k is not None:
#         rag_settings.RETRIEVAL_K = config.retrieval_k
#         updated_fields.append(f"retrieval_k → {config.retrieval_k}")

#     if config.fetch_k is not None:
#         rag_settings.FETCH_K = config.fetch_k
#         updated_fields.append(f"fetch_k → {config.fetch_k}")

#     if config.lambda_mult is not None:
#         rag_settings.LAMBDA_MULT = config.lambda_mult
#         updated_fields.append(f"lambda_mult → {config.lambda_mult}")

#     if config.similarity_threshold is not None:
#         rag_settings.SIMILARITY_THRESHOLD = config.similarity_threshold
#         updated_fields.append(f"similarity_threshold → {config.similarity_threshold}")

#     # ── Generation Settings ──
#     if config.llm_temperature is not None:
#         rag_settings.LLM_TEMPERATURE = config.llm_temperature
#         updated_fields.append(f"llm_temperature → {config.llm_temperature}")

#     if config.max_tokens is not None:
#         rag_settings.MAX_TOKENS = config.max_tokens
#         updated_fields.append(f"max_tokens → {config.max_tokens}")

#     # ── Document Processing ──
#     if config.chunk_size is not None:
#         rag_settings.CHUNK_SIZE = config.chunk_size
#         updated_fields.append(f"chunk_size → {config.chunk_size}")

#     if config.chunk_overlap is not None:
#         rag_settings.CHUNK_OVERLAP = config.chunk_overlap
#         updated_fields.append(f"chunk_overlap → {config.chunk_overlap}")

#     # Log changes
#     logger.info(f"📝 RAG Config Updated: {len(updated_fields)} field(s)")
#     for field in updated_fields:
#         logger.info(f"   • {field}")

#     # Return updated config
#     return await get_rag_config()


# @router.post("/upload_document", response_model=DocumentUploadResponse)
# async def upload_document(file: UploadFile = File(...), user_id: int = Depends(get_current_user_id)):
#     """
#     Upload a document to the RAG system and trigger ingestion.

#     This saves the file to the documents directory, updates the
#     active PDF path, and rebuilds the vector store.
#     """
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")

#     # Ensure documents directory exists
#     doc_dir = Path("documents")
#     doc_dir.mkdir(exist_ok=True)

#     file_path = doc_dir / file.filename

#     try:
#         # Save the uploaded file
#         with file_path.open("wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
            
#         logger.info(f"Document upload by user {user_id}: {file.filename}")
#         logger.info(f"📁 File saved to {file_path}")

#         # Update settings in-memory
#         rag_settings.PDF_PATH = str(file_path)

#         # Trigger ingestion (blocking call run in threadpool)
#         success = await run_in_threadpool(ingest_pdf, str(file_path))

#         if success:
#             # Clear retriever cache so it uses the new vector store
#             RetrieverFactory()._retriever = None

#             return DocumentUploadResponse(
#                 message="Document uploaded and ingested successfully",
#                 filename=file.filename,
#                 success=True,
#             )
#         else:
#             return DocumentUploadResponse(
#                 message="File uploaded but ingestion failed. Check server logs.",
#                 filename=file.filename,
#                 success=False,
#             )

#     except Exception as e:
#         logger.error(f"❌ Error during document upload/ingestion: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
