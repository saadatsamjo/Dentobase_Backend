# app/main.py
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import routers
from app.users.auth_routers import router as auth_router
from app.RAGsystem.routes import router as rag_router
from app.visionsystem.routes import router as vision_router
from app.cdss_engine.routes import router as cdss_router
from app.system_services.system_routes import router as system_router

# Import configurations
from config.ragconfig import rag_settings
from config.visionconfig import vision_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("\n")
    print("\n===============================================================================")
    print("===============================================================================")
    print(f" 🚀 Starting CDSS Server")
    print(f" ✅ Vision Model: {vision_settings.VISION_MODEL_PROVIDER} - {vision_settings.current_llm_model}")
    print(f" ✅ RAG Embeddings Provider: {rag_settings.EMBEDDING_PROVIDER} - {rag_settings.current_embedding_model}")
    print(f" ✅ Chunk Size: {rag_settings.CHUNK_SIZE}")
    print(f" ✅ Chunk Overlap: {rag_settings.CHUNK_OVERLAP}")
    print(f" ✅ LLM Provider: {rag_settings.LLM_PROVIDER} - {rag_settings.current_llm_model}")
    print(f" ✅ Retrieval Top-K: {rag_settings.RETRIEVAL_K}")
    print(f" ✅ Retrival Type: {rag_settings.RETRIEVER_TYPE}")
    if rag_settings.RETRIEVER_TYPE == "similarity_score_threshold":
        print(f" ✅ Similarity Threshold: {rag_settings.SIMILARITY_THRESHOLD}")
    if rag_settings.RETRIEVER_TYPE == "mmr":
        print(f" ✅ Fetch K: {rag_settings.FETCH_K}")
        print(f" ✅ Lambda Multiplier: {rag_settings.LAMBDA_MULT}")
    if rag_settings.EMBEDDING_PROVIDER == "huggingface":
        print(f" ✅ Computing Device: {rag_settings.effective_device}")
    print("===============================================================================")
    print("===============================================================================\n")
    yield
    # Shutdown
    print("👋 Shutting down")

app = FastAPI(
    title="Clinical Decision Support System",
    description="Multimodal CDSS with Florence-2 vision and RAG-based clinical guidelines",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers with prefixes
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(rag_router, prefix="/api/rag", tags=["Legacy RAG"])
app.include_router(vision_router, prefix="/api/vision", tags=["Vision Analysis"])
app.include_router(cdss_router, prefix="/api/cdss", tags=["Clinical Decision Support"])
app.include_router(system_router, prefix="/api/system", tags=["System Services"])



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

