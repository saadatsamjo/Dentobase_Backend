# app/RAGsystem/embeddings.py

from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config.ragconfig import rag_settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """Factory for embedding models with automatic provider switching."""
    
    _instance = None
    _embeddings = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_embeddings(self):
        """Get or initialize embeddings based on configuration."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def _create_embeddings(self):
        provider = rag_settings.EMBEDDING_PROVIDER
        model_name = rag_settings.current_embedding_model
        
        logger.info(f"Initializing {provider} embeddings: {model_name}")
        
        if provider == "ollama":
            return OllamaEmbeddings(model=model_name)
        
        elif provider == "huggingface":
            device = rag_settings.effective_device
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32 if device == "cuda" else 8
                }
            )
        
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    def refresh(self):
        """Force re-initialization (e.g., after config change)."""
        self._embeddings = None
        return self.get_embeddings()

# Global instance
embedding_provider = EmbeddingProvider()