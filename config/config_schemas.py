# config/config_schemas.py
"""
Configuration Schemas - COMPLETE WITH ALL MODELS
Includes: Groq, Gemini, Gemma3 support
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional


# ============================================================================
# VISION SYSTEM CONFIGURATION
# ============================================================================
class VisionConfigRequest(BaseModel):
    """Request to update vision system configuration."""
    
    # Model Selection
    vision_model_provider: Optional[Literal[
        "llava",        # Ollama: llava:13b, llama3.2-vision
        "gemma3",       # Ollama: gemma3:4b, gemma3:12b (NEW)
        "llava_med",    # HuggingFace: Medical specialist
        "biomedclip",   # HuggingFace: Pathology classifier
        "florence",     # HuggingFace: General vision
        "gpt4v",        # OpenAI: Premium cloud
        "claude",       # Anthropic: Premium cloud
        "groq",         # Groq: Ultra-fast cloud (NEW)
        "gemini"        # Google: Multimodal cloud (NEW)
    ]] = Field(None, description="Vision model provider")
    
    llava_model: Optional[str] = Field(
        None, 
        description="Ollama LLaVA model",
        examples=["llava:13b", "llama3.2-vision", "llava:latest"]
    )
    
    gemma3_model: Optional[str] = Field(
        None,
        description="Ollama Gemma3 model (NEW)",
        examples=["gemma3:4b", "gemma3:12b"]
    )
    
    groq_vision_model: Optional[str] = Field(
        None,
        description="Groq vision model (NEW)",
        examples=["meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct"]
    )
    
    gemini_vision_model: Optional[str] = Field(
        None,
        description="Google Gemini vision model (NEW)",
        examples=["gemini-2.0-flash", "gemini-1.5-pro"]
    )
    
    # Inference Settings
    vision_temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Temperature for vision model generation"
    )
    vision_max_tokens: Optional[int] = Field(
        None, ge=100, le=4000, description="Maximum tokens in vision response"
    )
    
    # Prompt Engineering
    include_clinical_notes_in_vision_model_prompt: Optional[bool] = Field(
        None, description="Include clinical notes in vision prompt"
    )
    
    # Image Processing
    enhance_contrast: Optional[bool] = Field(
        None, description="Apply contrast enhancement to X-rays"
    )
    contrast_factor: Optional[float] = Field(
        None, ge=1.0, le=3.0, description="Contrast enhancement factor"
    )
    brightness_factor: Optional[float] = Field(
        None, ge=0.5, le=2.0, description="Brightness adjustment factor"
    )


class VisionConfigResponse(BaseModel):
    """Current vision system configuration."""
    
    # Model Selection
    vision_model_provider: str
    current_vision_model: str
    llava_model: str
    gemma3_model: str  # NEW
    groq_vision_model: str  # NEW
    gemini_vision_model: str  # NEW
    openai_vision_model:str #NEW
    claude_vision_model:str #NEW
    florence_model_name:str #NEW
    llava_med_model:str #NEW
    biomedclip_model:str #NEW
    
    # Inference Settings
    vision_temperature: float
    vision_max_tokens: int
    
    # Prompt Engineering
    include_clinical_notes_in_vision_model_prompt: bool
    
    # Image Processing
    enhance_contrast: bool
    contrast_factor: float
    brightness_factor: float


# ============================================================================
# RAG SYSTEM CONFIGURATION
# ============================================================================
class RAGConfigRequest(BaseModel):
    """Request to update RAG system configuration."""
    
    # LLM Provider and Models
    llm_provider: Optional[Literal[
        "ollama",   # Local: Free
        "openai",   # Cloud: Premium
        "claude",   # Cloud: Premium
        "groq",     # Cloud: Ultra-fast (NEW)
        "gemini"    # Cloud: Google (NEW)
    ]] = Field(None, description="LLM provider for generating recommendations")
    
    ollama_llm_model: Optional[str] = Field(
        None,
        description="Ollama model name",
        examples=["llama3.1:8b", "mixtral:8x7b", "gemma3:4b"]
    )
    openai_llm_model: Optional[str] = Field(
        None,
        description="OpenAI model name",
        examples=["gpt-4o", "gpt-4-turbo-preview"]
    )
    claude_llm_model: Optional[str] = Field(
        None,
        description="Claude model name",
        examples=["claude-3-5-sonnet-20241022"]
    )
    groq_llm_model: Optional[str] = Field(
        None,
        description="Groq model name (NEW)",
        examples=["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    )
    gemini_llm_model: Optional[str] = Field(
        None,
        description="Gemini model name (NEW)",
        examples=["gemini-2.0-flash-exp", "gemini-1.5-pro"]
    )
    
    # Embedding Provider and Models
    embedding_provider: Optional[Literal["ollama", "huggingface"]] = Field(
        None, description="Embedding provider for document retrieval"
    )
    ollama_embedding_model: Optional[str] = Field(
        None,
        description="Ollama embedding model",
        examples=["nomic-embed-text", "mxbai-embed-large"]
    )
    hf_embedding_model: Optional[str] = Field(
        None,
        description="HuggingFace embedding model",
        examples=["abhinand/MedEmbed-large-v0.1", "BAAI/bge-large-en-v1.5"]
    )
    
    # Retrieval Settings
    retriever_type: Optional[Literal["similarity", "mmr", "multi_query", "similarity_score_threshold"]] = Field(
        None, description="Retrieval strategy"
    )
    retrieval_k: Optional[int] = Field(
        None, ge=1, le=20, description="Number of document chunks to retrieve"
    )
    fetch_k: Optional[int] = Field(
        None, ge=5, le=50, description="MMR: Number of candidates to consider"
    )
    lambda_mult: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="MMR: 1.0=relevance, 0.0=diversity"
    )
    similarity_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    
    # LLM Generation Settings
    llm_temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Temperature"
    )
    max_tokens: Optional[int] = Field(
        None, ge=100, le=4000, description="Maximum tokens in LLM response"
    )
    
    # Document Processing
    chunk_size: Optional[int] = Field(
        None, ge=500, le=5000, description="Text chunk size"
    )
    chunk_overlap: Optional[int] = Field(
        None, ge=0, le=500, description="Overlap between chunks"
    )


class RAGConfigResponse(BaseModel):
    """Current RAG system configuration."""
    
    # LLM
    llm_provider: str
    current_llm_model: str
    ollama_llm_model: str
    openai_llm_model: str
    claude_llm_model: str
    groq_llm_model: str  # NEW
    gemini_llm_model: str  # NEW
    
    # Embedding
    embedding_provider: str
    current_embedding_model: str
    ollama_embedding_model: str
    hf_embedding_model: str
    
    # Retrieval
    retriever_type: str
    retrieval_k: int
    fetch_k: int
    lambda_mult: float
    similarity_threshold: Optional[float]
    
    # Generation
    llm_temperature: float
    max_tokens: int
    
    # Processing
    chunk_size: int
    chunk_overlap: int
    pdf_path: str
    persist_dir: str
    device: str


# ============================================================================
# COMBINED SYSTEM STATUS
# ============================================================================
class SystemConfigResponse(BaseModel):
    """Complete system configuration (RAG + Vision combined)."""
    rag: RAGConfigResponse
    vision: VisionConfigResponse
    


