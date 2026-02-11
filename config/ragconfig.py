# config/ragconfig.py
"""
RAG System Configuration
Controls knowledge retrieval and LLM settings for clinical recommendations
"""
from pydantic_settings import BaseSettings
from typing import Literal, Optional
import torch

class RAGSettings(BaseSettings):
    """Configuration for Clinical Decision Support RAG System"""
    
    # ============================================================================
    # LLM SELECTION (for generating clinical recommendations)
    # ============================================================================
    # Which LLM to use for final recommendation generation
    # Options: "ollama", "openai", "claude"
    LLM_PROVIDER: Literal["ollama", "openai", "claude"] = "ollama"
    
    # Ollama LLM Model (if LLM_PROVIDER = "ollama")
    OLLAMA_LLM_MODEL: str = "llama3:8b"
    # Alternatives: "llama3:70b", "mistral", "mixtral:8x7b"
    
    # OpenAI Model (if LLM_PROVIDER = "openai")
    OPENAI_LLM_MODEL: str = "gpt-4-turbo-preview"
    # Alternatives: "gpt-4", "gpt-3.5-turbo"
    
    # Claude Model (if LLM_PROVIDER = "claude")
    CLAUDE_LLM_MODEL: str = "claude-3-5-sonnet-20241022"
    # Alternatives: "claude-3-opus-20240229", "claude-3-haiku-20240307"
    
    # ============================================================================
    # EMBEDDING MODEL SELECTION (for document retrieval)
    # ============================================================================
    # Which embedding provider to use
    # Options: "ollama", "huggingface"
    EMBEDDING_PROVIDER: Literal["ollama", "huggingface"] = "ollama"
    
    # Ollama Embedding Model (if EMBEDDING_PROVIDER = "ollama")
    # OLLAMA_EMBEDDING_MODEL: str =    "nomic-embed-text"  
    OLLAMA_EMBEDDING_MODEL: str =    "mxbai-embed-large"  
    
    
    # HuggingFace Embedding Model (if EMBEDDING_PROVIDER = "huggingface")
    # HF_EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    HF_EMBEDDING_MODEL: str = "abhinand/MedEmbed-large-v0.1"
    # HF_EMBEDDING_MODEL: str = "neuml/pubmedbert-base-embeddings"
    
    
    # ============================================================================
    # RETRIEVAL SETTINGS
    # ============================================================================
    # Which retrieval strategy to use
    # Options: "similarity", "mmr", "multi_query", "similarity_score_threshold"
    # Recommended: "mmr" for comprehensive clinical coverage
    RETRIEVER_TYPE: Literal["similarity", "mmr", "multi_query", "similarity_score_threshold"] = "similarity_score_threshold"
    
    # Number of document chunks to retrieve
    RETRIEVAL_K: int = 8
    
    # Number of candidates to consider (MMR only)
    FETCH_K: int = 20
    
    # Diversity vs relevance trade-off (MMR only)
    # 0.0 = maximum diversity, 1.0 = maximum relevance
    LAMBDA_MULT: float = 0.7  # Favor relevance for clinical precision
    
    # Minimum similarity threshold (optional, None to disable)
    SIMILARITY_THRESHOLD: Optional[float] = 0.8
    
    # ============================================================================
    # LLM GENERATION SETTINGS
    # ============================================================================
    # Temperature (0 = deterministic, 1 = creative)
    LLM_TEMPERATURE: float = 0.0  # Use 0 for clinical consistency
    
    # Maximum tokens in LLM response
    MAX_TOKENS: int = 1500
    
    # Force JSON output (recommended for structured recommendations)
    FORCE_JSON_OUTPUT: bool = True
    
    # ============================================================================
    # DOCUMENT PROCESSING
    # ============================================================================
    # Text chunk size for document splitting
    CHUNK_SIZE: int = 512  # Optimized for clinical context
    
    # Overlap between chunks (maintains context continuity)
    CHUNK_OVERLAP: int = 100
    
    # Path to clinical guidelines PDF
    PDF_PATH: str = "./documents/stg_document2.pdf"
    
    # ============================================================================
    # VECTOR STORE
    # ============================================================================
    # Directory for persisted vector embeddings
    PERSIST_DIR: str = "chroma_db"
    
    # ============================================================================
    # DEVICE CONFIGURATION
    # ============================================================================
    # Compute device for HuggingFace models
    # Options: "auto", "cuda", "mps", "cpu"
    DEVICE: str = "auto"
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
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
        """Get the active embedding model based on provider."""
        if self.EMBEDDING_PROVIDER == "ollama":
            return self.OLLAMA_EMBEDDING_MODEL
        return self.HF_EMBEDDING_MODEL
    
    @property
    def current_llm_model(self) -> str:
        """Get the active LLM model based on provider."""
        if self.LLM_PROVIDER == "ollama":
            return self.OLLAMA_LLM_MODEL
        elif self.LLM_PROVIDER == "openai":
            return self.OPENAI_LLM_MODEL
        else:  # claude
            return self.CLAUDE_LLM_MODEL

rag_settings = RAGSettings()