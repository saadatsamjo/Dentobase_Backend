# config/visionconfig.py
"""
Vision System Configuration
Controls which vision model is used for dental radiograph analysis
"""
from pydantic_settings import BaseSettings
from typing import Literal

class VisionSettings(BaseSettings):
    """Configuration for Vision Models"""
    
    # ============================================================================
    # VISION MODEL SELECTION
    # ============================================================================
    # Choose which vision model to use for image analysis
    # Options: "llava", "gpt4v", "claude", "florence"
    # 
    # Recommendations:
    #   - "llava" (default): Free, local, good for development
    #   - "llava:13b": Better accuracy, requires 16GB RAM
    #   - "gpt4v": Best accuracy, costs ~$0.01/image
    #   - "claude": Excellent medical reasoning, ~$0.01/image
    #   - "florence": Fast but poor on medical images (NOT recommended)
    VISION_MODEL_PROVIDER: Literal["llava", "gpt4v", "claude", "florence"] = "llava"
    
    # ============================================================================
    # LLAVA SPECIFIC SETTINGS (if VISION_MODEL_PROVIDER = "llava")
    # ============================================================================
    # LLaVA Model Size Selection
    # Options: "llava" (7B), "llava:13b" (13B), "llava:34b" (34B)
    LLAVA_MODEL: str = "llava"  # Change to "llava:13b" for production
    
    # ============================================================================
    # FLORENCE SETTINGS (if VISION_MODEL_PROVIDER = "florence")
    # ============================================================================
    # NOT RECOMMENDED for dental X-rays - included for completeness
    FLORENCE_MODEL_NAME: str = "microsoft/Florence-2-base"
    # Alternative: "microsoft/Florence-2-large" (slower, marginally better)
    
    # ============================================================================
    # IMAGE PROCESSING SETTINGS
    # ============================================================================
    # Maximum image dimension (images resized to fit, maintaining aspect ratio)
    MAX_IMAGE_SIZE: int = 1024
    
    # Supported image formats
    SUPPORTED_FORMATS: list = ["image/jpeg", "image/png", "image/webp"]
    
    # Image enhancement for X-rays (LLaVA only)
    ENHANCE_CONTRAST: bool = True  # Improves X-ray visibility
    CONTRAST_FACTOR: float = 1.5   # 1.0 = no change, >1.0 = more contrast
    BRIGHTNESS_FACTOR: float = 1.2 # 1.0 = no change, >1.0 = brighter
    
    # ============================================================================
    # API SETTINGS (for GPT-4V and Claude)
    # ============================================================================
    # Request timeout for API calls (seconds)
    REQUEST_TIMEOUT: int = 30
    
    # Temperature for vision models (lower = more consistent)
    VISION_TEMPERATURE: float = 0.2
    
    # Maximum tokens for vision analysis
    VISION_MAX_TOKENS: int = 1500
    
    # ============================================================================
    # CLINICAL ANALYSIS SETTINGS
    # ============================================================================
    # Whether to perform dual-prompt analysis (detailed + pathology-focused)
    DUAL_PROMPT_ANALYSIS: bool = True
    
    # Minimum confidence threshold for reporting findings
    MIN_CONFIDENCE: float = 0.5
    
    
    # getting the current vision model
    @property
    def current_llm_model(self) -> str:
        """Get the active LLM model based on provider."""
        if self.VISION_MODEL_PROVIDER == "llava":
            return self.LLAVA_MODEL
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