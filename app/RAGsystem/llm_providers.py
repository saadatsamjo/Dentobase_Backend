# app/RAGsystem/llm_providers.py
"""
Additional LLM Provider Implementations for RAG System
Groq: Ultra-fast LPU inference
Gemini: Google's multimodal LLM with large context
"""
import logging
from typing import Dict, List
from config.ragconfig import rag_settings

logger = logging.getLogger(__name__)


def get_groq_llm():
    """
    Get Groq LLM for text generation.
    Uses Groq's LPU for 10-20x faster inference than GPU.
    """
    try:
        from langchain_groq import ChatGroq
        import os
        
        api_key = rag_settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        llm = ChatGroq(
            model=rag_settings.GROQ_LLM_MODEL,
            temperature=rag_settings.LLM_TEMPERATURE,
            max_tokens=rag_settings.MAX_TOKENS,
            groq_api_key=api_key,
        )
        
        logger.info(f"✅ Groq LLM initialized: {rag_settings.GROQ_LLM_MODEL}")
        return llm
    
    except ImportError:
        raise ImportError("langchain-groq not installed. Run: pip install langchain-groq")
    except Exception as e:
        raise RuntimeError(f"Groq LLM initialization failed: {e}")


def get_gemini_llm():
    """
    Get Gemini LLM for text generation.
    Supports up to 2M token context window.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        
        api_key = rag_settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        
        llm = ChatGoogleGenerativeAI(
            model=rag_settings.GEMINI_LLM_MODEL,
            temperature=rag_settings.LLM_TEMPERATURE,
            max_output_tokens=rag_settings.MAX_TOKENS,
            google_api_key=api_key,
        )
        
        logger.info(f"✅ Gemini LLM initialized: {rag_settings.GEMINI_LLM_MODEL}")
        return llm
    
    except ImportError:
        raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
    except Exception as e:
        raise RuntimeError(f"Gemini LLM initialization failed: {e}")


def get_llm_by_provider(provider: str = None):
    """
    Get LLM instance based on provider.
    Factory function for all LLM providers.
    
    Args:
        provider: Override rag_settings.LLM_PROVIDER
    
    Returns:
        LangChain LLM instance
    """
    provider = provider or rag_settings.LLM_PROVIDER
    
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=rag_settings.OLLAMA_LLM_MODEL,
            temperature=rag_settings.LLM_TEMPERATURE,
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=rag_settings.OPENAI_LLM_MODEL,
            temperature=rag_settings.LLM_TEMPERATURE,
            max_tokens=rag_settings.MAX_TOKENS,
        )
    
    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=rag_settings.CLAUDE_LLM_MODEL,
            temperature=rag_settings.LLM_TEMPERATURE,
            max_tokens=rag_settings.MAX_TOKENS,
        )
    
    elif provider == "groq":
        return get_groq_llm()
    
    elif provider == "gemini":
        return get_gemini_llm()
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")