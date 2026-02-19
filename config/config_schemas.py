# config/config_schemas.py
"""
Unified Configuration Schemas for RAG and Vision Systems
Used by: /api/rag/config, /api/vision/config

Design: In-memory configuration (no database persistence)
- GET /config → returns current settings
- POST /config → updates settings in-memory (partial updates supported)
- Settings reset to file defaults on application restart
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# RAG SYSTEM CONFIGURATION
# ============================================================================
class RAGConfigRequest(BaseModel):
    """
    Request to update RAG configuration.
    All fields are optional — send only what you want to change.
    """

    # ── LLM Provider and Models ──
    llm_provider: Optional[Literal["ollama", "openai", "claude"]] = Field(
        None, description="LLM provider for generating clinical recommendations"
    )
    ollama_llm_model: Optional[Literal["llama3.1:8b", "mixtral:8x7b", "llama3:8b"]] = Field(
        None,
        description="Ollama model name (e.g., 'llama3.1:8b', 'mixtral:8x7b')",
    )
    openai_llm_model: Optional[str] = Field(
        None, description="OpenAI model name", examples=["gpt-4-turbo-preview", "gpt-4o"]
    )
    claude_llm_model: Optional[str] = Field(
        None, description="Claude model name", examples=["claude-3-5-sonnet-20241022"]
    )

    # ── Embedding Provider and Models ──
    embedding_provider: Optional[Literal["ollama", "huggingface"]] = Field(
        None, 
        description="Embedding provider for document retrieval"
    )
    ollama_embedding_model: Optional[Literal["nomic-embed-text", "mxbai-embed-large"]] = Field(
        None,
        description="Ollama embedding model",
    )
    hf_embedding_model: Optional[Literal["abhinand/MedEmbed-large-v0.1", "BAAI/bge-large-en-v1.5", "neuml/pubmedbert-base-embeddings"]] = Field(
        None,
        description="HuggingFace embedding model",
    )

    # ── Retrieval Strategy ──
    retriever_type: Optional[
        Literal["similarity", "mmr", "multi_query", "similarity_score_threshold"]
    ] = Field(
        None,
        description="Retrieval strategy: mmr (recommended), similarity, multi_query, similarity_score_threshold",
    )
    retrieval_k: Optional[int] = Field(
        None, ge=1, le=20, description="Number of document chunks to retrieve (1-20)"
    )
    fetch_k: Optional[int] = Field(
        None,
        ge=5,
        le=50,
        description="MMR only: Number of candidates to consider before diversity selection (must be >= retrieval_k)",
    )
    lambda_mult: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="MMR only: Trade-off between relevance (1.0) and diversity (0.0)",
    )
    similarity_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity score (0.0-1.0, or null to disable)"
    )

    # ── LLM Generation Settings ──
    llm_temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Temperature: 0.0=deterministic (recommended for clinical), 1.0=balanced, 2.0=creative",
    )
    max_tokens: Optional[int] = Field(
        None, ge=100, le=4000, description="Maximum tokens in LLM response"
    )

    # ── Document Processing ──
    chunk_size: Optional[int] = Field(
        None, ge=500, le=5000, description="Text chunk size for document splitting (characters)"
    )
    chunk_overlap: Optional[int] = Field(
        None, ge=0, le=500, description="Overlap between chunks to maintain context"
    )


class RAGConfigResponse(BaseModel):
    """Current RAG system configuration (complete state)."""

    # LLM
    llm_provider: str
    current_llm_model: str  # Computed: the active model based on provider
    ollama_llm_model: str
    openai_llm_model: str
    claude_llm_model: str

    # Embedding
    embedding_provider: str
    current_embedding_model: str  # Computed: the active embedding model
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

    # Paths (read-only)
    pdf_path: str
    persist_dir: str
    device: str  # Computed: effective device (cuda/mps/cpu)


# ============================================================================
# VISION SYSTEM CONFIGURATION
# ============================================================================
class VisionConfigRequest(BaseModel):
    """
    Request to update vision configuration.
    All fields are optional — send only what you want to change.
    """

    # ── Model Selection ──
    vision_model_provider: Optional[
        Literal["florence", "llava", "llava_med", "biomedclip", "gpt4v", "claude"]
    ] = Field(None, description="Vision model provider")
    llava_model: Optional[Literal["llava:13b", "llama3.2-vision", "llava:latest"]] = Field(
        None,
        description="Ollama LLaVA model name",
    )

    # ── Inference Settings ──
    vision_temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Temperature for vision model text generation (0.0 recommended)",
    )
    vision_max_tokens: Optional[int] = Field(
        None, ge=100, le=4000, description="Maximum tokens in vision response"
    )

    # ── Prompt Engineering ──
    include_clinical_notes_in_vision_model_prompt: Optional[bool] = Field(
        None, description="Include clinical notes in vision model prompt (recommended: True)"
    )

    # ── Image Processing ──
    enhance_contrast: Optional[bool] = Field(
        None, description="Apply contrast enhancement to X-ray images (experimental)"
    )
    contrast_factor: Optional[float] = Field(
        None, ge=1.0, le=3.0, description="Contrast enhancement multiplier (1.0-3.0)"
    )
    brightness_factor: Optional[float] = Field(
        None, ge=0.5, le=2.0, description="Brightness adjustment multiplier (0.5-2.0)"
    )


class VisionConfigResponse(BaseModel):
    """Current vision system configuration (complete state)."""

    # Model Selection
    vision_model_provider: str
    current_vision_model: str  # Computed: the active vision model
    llava_model: str | None

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
# COMBINED SYSTEM STATUS (Optional — for dashboard)
# ============================================================================
class SystemConfigResponse(BaseModel):
    """Complete system configuration (RAG + Vision combined)."""

    rag: RAGConfigResponse
    vision: VisionConfigResponse

    class Config:
        json_schema_extra = {
            "example": {
                "rag": {
                    "llm_provider": "ollama",
                    "current_llm_model": "llama3.1:8b",
                    "retriever_type": "mmr",
                    "retrieval_k": 8,
                },
                "vision": {
                    "vision_model_provider": "llava",
                    "current_vision_model": "llava:13b",
                    "vision_temperature": 0.0,
                },
            }
        }
