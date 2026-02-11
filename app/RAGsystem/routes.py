# app/RAGsystem/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.RAGsystem.chains import ClinicalRAGChain, answer_question
from app.RAGsystem.retriever import get_retriever, RetrieverFactory
from config.ragconfig import rag_settings

router = APIRouter()

class QuestionPayload(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    retrieval_strategy: str
    
class ProviderSwitchRequest(BaseModel):
    provider: str
    model: str = None

@router.post("/answer_question", response_model=RAGResponse)
async def answer_question_endpoint(payload: QuestionPayload):
    """
    Legacy RAG endpoint - direct question answering.
    For full CDSS use /api/cdss/provide_final_recommendation
    """
    if not payload.question:
        raise HTTPException(status_code=400, detail="Question is required")

    retriever = RetrieverFactory().get_retriever()
    chain = ClinicalRAGChain(retriever)
    answer = chain.invoke(payload.question)
    
    return RAGResponse(
        answer=answer,
        retrieval_strategy=rag_settings.RETRIEVER_TYPE
    )
    
    
@router.post("/switch_provider")
async def switch_embedding_provider(request: ProviderSwitchRequest):
    """
    Dynamically switch between Ollama and HuggingFace embeddings.
    """
    from app.RAGsystem.embeddings import embedding_provider
    
    if request.provider not in ["ollama", "huggingface"]:
        raise HTTPException(
            status_code=400, 
            detail="Provider must be 'ollama' or 'huggingface'"
        )
    
    # Update settings
    rag_settings.EMBEDDING_PROVIDER = request.provider
    if request.model:
        if request.provider == "ollama":
            rag_settings.OLLAMA_EMBEDDING_MODEL = request.model
        else:
            rag_settings.HF_EMBEDDING_MODEL = request.model
    
    # Refresh embeddings
    embedding_provider.refresh()
    
    return {
        "message": f"Switched to {request.provider}",
        "current_model": rag_settings.current_embedding_model,
        "note": "You may need to re-ingest documents for optimal performance"
    }