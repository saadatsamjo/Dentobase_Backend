# config/visionconfig.py
"""
Vision Model Configuration - COMPLETE AND ORGANIZED
Supports: LLaVA (7B/13B/Llama3.2), LLaVA-Med, GPT-4V, Claude, Florence, BiomedCLIP
"""
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class VisionSettings(BaseSettings):
    """Configuration for all vision models in the system"""
    
    # ========================================================================
    # PRIMARY VISION MODEL SELECTION
    # ========================================================================
    # Choose which model to use for analysis
    # Options: "llava", "llava_med", "biomedclip", "gpt4v", "claude", "florence"
    #
    # RECOMMENDATIONS (based on your ollama list):
    #   LOCAL MODELS (Free, No API):
    #   - "llava" with LLAVA_MODEL="llava:13b" → Best open-source option
    #   - "llava" with LLAVA_MODEL="llama3.2-vision" → Latest from Meta
    #   - "llava_med" → Medical-specific (HuggingFace transformers)
    #   - "biomedclip" → Medical classifier (good for pathology detection)
    #
    #   PROPRIETARY (API costs):
    #   - "gpt4v" → Best accuracy ($0.01/image)
    #   - "claude" → Excellent reasoning ($0.01/image)
    #
    #   NOT RECOMMENDED:
    #   - "florence" → Poor on dental X-rays
    
    VISION_MODEL_PROVIDER: Literal["florence", "llava", "llava_med", "biomedclip", "gpt4v", "claude"] = "biomedclip"
    
    
    # ========================================================================
    # LLAVA SETTINGS (Ollama models)
    # ========================================================================
    # Your available models from `ollama list`:
    # - llava:latest - Fast but less accurate
    # - llava:13b - RECOMMENDED for production
    # - llama3.2-vision - Latest from Meta, good performance
    
    # LLAVA_MODEL: str = "llava:latest" 
    # LLAVA_MODEL: str = "llava:13b" 
    LLAVA_MODEL: str = "llama3.2-vision" 
    
    
    # ========================================================================
    # LLAVA-MED SETTINGS (HuggingFace transformers)
    # ========================================================================
    # Medical-specific vision model via transformers library
    # Downloads model from HuggingFace on first use
    # Available models from your search:
    # - mradermacher/llava-med-v1.5-mistral-7b-GGUF (recommended)
    # - sbottazzi/LLaVA-Med_weights_gguf
    
    # LLAVA_MED_MODEL: str = "sbottazzi/LLaVA-Med_weights_gguf"
    # LLAVA_MED_MODEL: str = "microsoft/llava-med-v1.5-mistral-7b"
    LLAVA_MED_MODEL: str = "mradermacher/llava-med-v1.5-mistral-7b-GGUF"
    LLAVA_MED_DEVICE: str = "cpu"
    
    
    # ========================================================================
    # BIOMEDCLIP SETTINGS (HuggingFace transformers)
    # ========================================================================
    # Medical image classifier trained on 15M medical images
    # Good for pathology detection
    
    BIOMEDCLIP_MODEL: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    BIOMEDCLIP_DEVICE: str = "cpu"
    
    
    # ========================================================================
    # FLORENCE SETTINGS (HuggingFace - NOT RECOMMENDED)
    # ========================================================================
    # You have these cached already:
    # - microsoft/Florence-2-base (465MB)
    # - microsoft/Florence-2-large (1.6GB)
    
    # FLORENCE_MODEL_NAME: str = "microsoft/Florence-2-base"
    FLORENCE_MODEL_NAME: str = "microsoft/Florence-2-large"
    
    
    # ========================================================================
    # PROPRIETARY API KEYS
    # ========================================================================
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
    
    
    # ========================================================================
    # IMAGE PROCESSING
    # ========================================================================
    MAX_IMAGE_SIZE: int = 1024  # Max dimension
    SUPPORTED_FORMATS: list = ["image/jpeg", "image/png", "image/webp"]
    
    # X-ray enhancement (only for LLaVA)
    ENHANCE_CONTRAST: bool = False
    CONTRAST_FACTOR: float = 1.5
    BRIGHTNESS_FACTOR: float = 1.2
    
    
    # ========================================================================
    # ANALYSIS SETTINGS
    # ========================================================================
    VISION_TEMPERATURE: float = 0.0  # Deterministic for clinical use
    VISION_MAX_TOKENS: int = 1500
    REQUEST_TIMEOUT: int = 30
    
    # Whether to include clinical notes in vision prompt
    # (disabled for Florence due to token limits)
    INCLUDE_CLINICAL_NOTES_IN_VISION_MODEL_PROMPT: bool = True
    
    # Dual-prompt analysis (detailed + pathology-focused)
    DUAL_PROMPT_ANALYSIS: bool = True
    
    
    # ========================================================================
    # HELPER PROPERTIES
    # ========================================================================
    @property
    def current_vision_model(self) -> str:
        """Get the active model name for display"""
        if self.VISION_MODEL_PROVIDER == "llava":
            return self.LLAVA_MODEL
        elif self.VISION_MODEL_PROVIDER == "llava_med":
            return self.LLAVA_MED_MODEL
        elif self.VISION_MODEL_PROVIDER == "biomedclip":
            return self.BIOMEDCLIP_MODEL
        elif self.VISION_MODEL_PROVIDER == "gpt4v":
            return "gpt-4-vision-preview"
        elif self.VISION_MODEL_PROVIDER == "claude":
            return "claude-3-5-sonnet-20241022"
        else:  # florence
            return self.FLORENCE_MODEL_NAME
    
    class Config:
        env_file = ".env"
        extra = "ignore"


vision_settings = VisionSettings()