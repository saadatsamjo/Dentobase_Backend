# config/ragconfig.py
"""
RAG System Configuration - COMPLETE WITH ALL LLM PROVIDERS
Controls knowledge retrieval and LLM settings for clinical recommendations
Supports: Ollama, OpenAI, Claude, Groq, Gemini
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional
import torch
import os
from pathlib import Path

# Calculate the project root
BASE_DIR = Path(__file__).resolve().parent.parent


class RAGSettings(BaseSettings):
    """Configuration for Clinical Decision Support RAG System"""
    
    # ============================================================================
    # LLM SELECTION (for generating clinical recommendations)
    # ============================================================================
    LLM_PROVIDER: Literal["ollama", "openai", "claude", "groq", "gemini"] = "ollama"
    
    # ── Ollama LLM Settings (Local, Free) ──
    OLLAMA_LLM_MODEL: str = "llama3.1:8b"
    # Alternatives: "llama3:8b", "mixtral:8x7b", "gemma3:4b"
    
    # ── OpenAI Settings (Cloud, Paid) ──
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_LLM_MODEL: str = "gpt-4o"
    # Alternatives: "gpt-4-turbo-preview", "gpt-3.5-turbo"
    
    # ── Claude Settings (Cloud, Paid) ──
    CLAUDE_API_KEY: str = Field(default="", env="CLAUDE_API_KEY")
    CLAUDE_LLM_MODEL: str = "claude-3-5-sonnet-20241022"
    # Alternatives: "claude-3-opus-20240229", "claude-3-haiku-20240307"
    
    # ── Groq Settings (NEW - Cloud, Ultra-fast, Free tier) ──
    # Free tier: 7,000 requests/day!
    # Best for: Speed-critical applications
    GROQ_API_KEY: str = Field(default="", env="GROQ_API_KEY")
    GROQ_LLM_MODEL: str = "llama-3.3-70b-versatile"
    # Alternatives: "mixtral-8x7b-32768", "llama-3.1-70b-versatile"
    
    # ── Gemini Settings (NEW - Google Cloud, Competitive pricing) ──
    # Free tier: 1,500 requests/day
    # Best for: Long context (2M tokens), multimodal reasoning
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    GEMINI_LLM_MODEL: str = "gemini-2.0-flash-exp"
    # Alternatives: "gemini-1.5-pro", "gemini-1.5-flash"

    # ============================================================================
    # EMBEDDING MODEL SELECTION (for document retrieval)
    # ============================================================================
    EMBEDDING_PROVIDER: Literal["ollama", "huggingface"] = "ollama"
    
    # ── Ollama Embedding Settings ──
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    # Alternatives: "mxbai-embed-large"
    
    # ── HuggingFace Embedding Settings ──
    HF_EMBEDDING_MODEL: str = "neuml/pubmedbert-base-embeddings"
    # Alternatives: "abhinand/MedEmbed-large-v0.1", "BAAI/bge-large-en-v1.5"

    # ============================================================================
    # RETRIEVAL SETTINGS
    # ============================================================================
    RETRIEVER_TYPE: Literal["similarity", "mmr", "multi_query", "similarity_score_threshold"] = "mmr"
    RETRIEVAL_K: int = 8              # Number of chunks to retrieve
    FETCH_K: int = 20                 # MMR: Candidates to consider
    LAMBDA_MULT: float = 1.0          # MMR: 1.0=relevance, 0.0=diversity
    SIMILARITY_THRESHOLD: Optional[float] = 0.6  # Min similarity score

    # ============================================================================
    # LLM GENERATION SETTINGS
    # ============================================================================
    LLM_TEMPERATURE: float = 0.0      # 0=deterministic (clinical use)
    MAX_TOKENS: int = 1500
    FORMAT: Literal["json", "markdown"] = "json"
    FORCE_JSON_OUTPUT: bool = True

    # ============================================================================
    # DOCUMENT PROCESSING
    # ============================================================================
    CHUNK_SIZE: int = 1500           # Characters per chunk
    CHUNK_OVERLAP: int = 100         # Overlap for context continuity
    PDF_PATH: str = str(BASE_DIR / "documents" / "stg.pdf")

    # ============================================================================
    # VECTOR STORE
    # ============================================================================
    PERSIST_DIR: str = str(BASE_DIR / "chroma_db")

    # ============================================================================
    # DEVICE CONFIGURATION
    # ============================================================================
    DEVICE: str = "auto"  # Options: "auto", "cuda", "mps", "cpu"

    class Config:
        env_file = ".env"
        extra = "ignore"
    
    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================
    @property
    def resolved_pdf_path(self) -> Path:
        """Get absolute path to PDF document."""
        path = Path(self.PDF_PATH)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def resolved_persist_dir(self) -> Path:
        """Get absolute path to vector store directory."""
        path = Path(self.PERSIST_DIR)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def effective_device(self) -> str:
        """Determine compute device based on configuration."""
        if self.DEVICE == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.DEVICE
    
    @property
    def current_embedding_model(self) -> str:
        """Get active embedding model based on provider."""
        if self.EMBEDDING_PROVIDER == "ollama":
            return self.OLLAMA_EMBEDDING_MODEL
        return self.HF_EMBEDDING_MODEL
    
    @property
    def current_llm_model(self) -> str:
        """Get active LLM model based on provider."""
        provider_map = {
            "ollama": self.OLLAMA_LLM_MODEL,
            "openai": self.OPENAI_LLM_MODEL,
            "claude": self.CLAUDE_LLM_MODEL,
            "groq": self.GROQ_LLM_MODEL,
            "gemini": self.GEMINI_LLM_MODEL,
        }
        return provider_map.get(self.LLM_PROVIDER, self.OLLAMA_LLM_MODEL)


rag_settings = RAGSettings()