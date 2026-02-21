# config/visionconfig.py
"""
Vision Model Configuration - COMPLETE WITH ALL MODELS
Supports: 
  LOCAL: LLaVA, Gemma3, LLaVA-Med, BiomedCLIP, Florence
  CLOUD: GPT-4V, Claude, Groq, Gemini
"""
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class VisionSettings(BaseSettings):
    """Configuration for all vision models in the system"""

    # ========================================================================
    # PRIMARY VISION MODEL SELECTION
    # ========================================================================
    VISION_MODEL_PROVIDER: Literal[
        "llava",        # Ollama: llava:13b, llama3.2-vision, llava:latest
        "gemma3",       # Ollama: gemma3:4b, gemma3:12b 
        "llava_med",    # HuggingFace: Medical specialist
        "biomedclip",   # HuggingFace: Pathology classifier
        "florence",     # HuggingFace: General vision (not recommended)
        "gpt4v",        # OpenAI: Best accuracy, expensive
        "claude",       # Anthropic: Excellent reasoning
        "groq",         # Groq: Ultra-fast, cheap 
        "gemini"        # Google: Multimodal, large context
    ] = "gemma3"

    # ========================================================================
    # LOCAL MODELS - OLLAMA (Free, No API)
    # ========================================================================
    
    # ── LLaVA Settings ──
    # Available: llava:13b (recommended), llama3.2-vision (Meta latest), llava:latest
    LLAVA_MODEL: str = "llava:13b"
    
    # ── Gemma 3 Settings (NEW - Google's multimodal) ──
    # Available: gemma3:4b (recommended for 16GB RAM), gemma3:12b (better accuracy)
    # Note: gemma3:4b is vision-capable, gemma3:1b is text-only
    GEMMA3_MODEL: str = "gemma3:4b"
    
    # ── LLaVA-Med Settings (HuggingFace transformers) ──
    # Medical-specific vision model
    LLAVA_MED_MODEL: str = "mradermacher/llava-med-v1.5-mistral-7b-GGUF"
    LLAVA_MED_DEVICE: str = "cpu"
    
    # ── BiomedCLIP Settings (HuggingFace open_clip) ──
    # Medical pathology classifier
    BIOMEDCLIP_MODEL: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    BIOMEDCLIP_DEVICE: str = "cpu"
    
    # ── Florence Settings (HuggingFace - not recommended for dental) ──
    FLORENCE_MODEL_NAME: str = "microsoft/Florence-2-large"

    # ========================================================================
    # CLOUD MODELS - API (Paid/Free Tier)
    # ========================================================================
    
    # ── OpenAI Settings ──
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_VISION_MODEL: str = "gpt-4o"  # or "gpt-4-vision-preview"
    
    # ── Anthropic Settings ──
    ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
    CLAUDE_VISION_MODEL: str = "claude-3-5-sonnet-20241022"
    
    # ── Groq Settings (NEW - Ultra-fast LPU inference) ──
    # Free tier: 7,000 requests/day!
    # Models: meta-llama/llama-4-maverick-17b-128e-instruct (best), meta-llama/llama-4-scout-17b-16e-instruct (fast)
    GROQ_API_KEY: str = Field(default="", env="GROQ_API_KEY")
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # or "meta-llama/llama-4-maverick-17b-128e-instruct" foe complex reasoning
    
    # ── Gemini Settings (NEW - Google's multimodal) ──
    # Free tier: 1,500 requests/day
    # Models: gemini-2.0-flash-exp (recommended), gemini-1.5-pro, gemini-1.5-flash
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    GEMINI_VISION_MODEL: str = "gemini-2.0-flash"  # Fast, cheap, good

    # ========================================================================
    # IMAGE PROCESSING
    # ========================================================================
    MAX_IMAGE_SIZE: int = 1024  # Max dimension for preprocessing
    SUPPORTED_FORMATS: list = ["image/jpeg", "image/png", "image/webp"]
    
    # X-ray contrast enhancement (experimental)
    ENHANCE_CONTRAST: bool = False
    CONTRAST_FACTOR: float = 1.5
    BRIGHTNESS_FACTOR: float = 1.2

    # ========================================================================
    # ANALYSIS SETTINGS
    # ========================================================================
    VISION_TEMPERATURE: float = 0.0  # Deterministic for clinical use
    VISION_MAX_TOKENS: int = 1500
    REQUEST_TIMEOUT: int = 30
    
    # Include clinical notes in vision prompt
    INCLUDE_CLINICAL_NOTES_IN_VISION_MODEL_PROMPT: bool = True
    
    # Dual-prompt analysis (detailed + pathology-focused)
    DUAL_PROMPT_ANALYSIS: bool = True

    # ========================================================================
    # HELPER PROPERTIES
    # ========================================================================
    @property
    def current_vision_model(self) -> str:
        """Get the active model name for display"""
        provider_map = {
            "llava": self.LLAVA_MODEL,
            "gemma3": self.GEMMA3_MODEL,
            "llava_med": self.LLAVA_MED_MODEL,
            "biomedclip": self.BIOMEDCLIP_MODEL,
            "florence": self.FLORENCE_MODEL_NAME,
            "gpt4v": self.OPENAI_VISION_MODEL,
            "claude": self.CLAUDE_VISION_MODEL,
            "groq": self.GROQ_VISION_MODEL,
            "gemini": self.GEMINI_VISION_MODEL,
        }
        return provider_map.get(self.VISION_MODEL_PROVIDER, "unknown")

    class Config:
        env_file = ".env"
        extra = "ignore"


vision_settings = VisionSettings()