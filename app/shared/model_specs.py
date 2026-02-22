# app/shared/model_specs.py

"""
Complete Model Specifications Database
Used for thesis tables and deployment analysis
All data verified as of February 2026
"""
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ModelSpecs:
    """Complete specifications for a vision or LLM model"""
    name: str
    display_name: str
    size_gb: Optional[float]  # None for cloud models
    params_billions: Optional[float]
    requires_gpu: bool
    min_ram_gb: int
    min_vram_gb: Optional[int]  # For GPU models
    quantization: Optional[str]
    provider: str  # "local" or "cloud"
    platform: str  # "ollama", "huggingface", "openai", "anthropic", "groq", "google"
    supports_vision: bool
    context_window: Optional[int]  # Tokens


# ============================================================================
# VISION MODELS SPECIFICATIONS
# ============================================================================

VISION_MODEL_SPECS: Dict[str, ModelSpecs] = {
    # ── Local Models (Ollama) ──
    "llava:13b": ModelSpecs(
        name="llava:13b",
        display_name="LLaVA 13B",
        size_gb=7.4,
        params_billions=13.0,
        requires_gpu=False,
        min_ram_gb=16,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=True,
        context_window=4096
    ),
    
    "llava:latest": ModelSpecs(
        name="llava:latest",
        display_name="LLaVA 7B",
        size_gb=4.7,
        params_billions=7.0,
        requires_gpu=False,
        min_ram_gb=8,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=True,
        context_window=4096
    ),
    
    "llama3.2-vision": ModelSpecs(
        name="llama3.2-vision",
        display_name="Llama 3.2 Vision 11B",
        size_gb=7.9,
        params_billions=11.0,
        requires_gpu=False,
        min_ram_gb=16,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=True,
        context_window=128000
    ),
    
    "gemma3:4b": ModelSpecs(
        name="gemma3:4b",
        display_name="Gemma 3 4B",
        size_gb=2.5,
        params_billions=4.0,
        requires_gpu=False,
        min_ram_gb=8,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=True,
        context_window=8192
    ),
    
    "gemma3:12b": ModelSpecs(
        name="gemma3:12b",
        display_name="Gemma 3 12B",
        size_gb=7.0,
        params_billions=12.0,
        requires_gpu=False,
        min_ram_gb=16,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=True,
        context_window=8192
    ),
    
    # ── Local Models (HuggingFace) ──
    "biomedclip": ModelSpecs(
        name="biomedclip",
        display_name="BiomedCLIP",
        size_gb=0.5,
        params_billions=None,
        requires_gpu=False,
        min_ram_gb=4,
        min_vram_gb=None,
        quantization=None,
        provider="local",
        platform="huggingface",
        supports_vision=True,
        context_window=77
    ),
    
    "florence": ModelSpecs(
        name="florence",
        display_name="Florence-2 Large",
        size_gb=1.6,
        params_billions=0.77,
        requires_gpu=False,
        min_ram_gb=8,
        min_vram_gb=None,
        quantization=None,
        provider="local",
        platform="huggingface",
        supports_vision=True,
        context_window=1024
    ),
    
    # ── Cloud Models ──
    "gpt-4o": ModelSpecs(
        name="gpt-4o",
        display_name="GPT-4o Vision",
        size_gb=None,
        params_billions=None,  # Not disclosed
        requires_gpu=False,
        min_ram_gb=4,  # Only for API client
        min_vram_gb=None,
        quantization=None,
        provider="cloud",
        platform="openai",
        supports_vision=True,
        context_window=128000
    ),
    
    "claude-3-5-sonnet-20241022": ModelSpecs(
        name="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        size_gb=None,
        params_billions=None,
        requires_gpu=False,
        min_ram_gb=4,
        min_vram_gb=None,
        quantization=None,
        provider="cloud",
        platform="anthropic",
        supports_vision=True,
        context_window=200000
    ),
    
    "llama-3.2-11b-vision-preview": ModelSpecs(
        name="llama-3.2-11b-vision-preview",
        display_name="Groq Llama 3.2 11B Vision",
        size_gb=None,
        params_billions=11.0,
        requires_gpu=False,
        min_ram_gb=4,
        min_vram_gb=None,
        quantization=None,
        provider="cloud",
        platform="groq",
        supports_vision=True,
        context_window=8192
    ),
    
    "gemini-1.5-flash": ModelSpecs(
        name="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        size_gb=None,
        params_billions=None,
        requires_gpu=False,
        min_ram_gb=4,
        min_vram_gb=None,
        quantization=None,
        provider="cloud",
        platform="google",
        supports_vision=True,
        context_window=1000000
    ),
}


# ============================================================================
# LLM MODELS SPECIFICATIONS (Text-only)
# ============================================================================

LLM_MODEL_SPECS: Dict[str, ModelSpecs] = {
    # ── Local ──
    "llama3.1:8b": ModelSpecs(
        name="llama3.1:8b",
        display_name="Llama 3.1 8B",
        size_gb=4.7,
        params_billions=8.0,
        requires_gpu=False,
        min_ram_gb=8,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=False,
        context_window=128000
    ),
    
    "mixtral:8x7b": ModelSpecs(
        name="mixtral:8x7b",
        display_name="Mixtral 8x7B",
        size_gb=26.0,
        params_billions=46.7,
        requires_gpu=False,
        min_ram_gb=32,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=False,
        context_window=32000
    ),
    
    "llama-3.3-70b-versatile": ModelSpecs(
        name="llama-3.3-70b-versatile",
        display_name="Groq Llama 3.3 70B",
        size_gb=None,
        params_billions=70.0,
        requires_gpu=False,
        min_ram_gb=4,
        min_vram_gb=None,
        quantization=None,
        provider="cloud",
        platform="groq",
        supports_vision=False,
        context_window=8192
    ),
    "gemma3": ModelSpecs(
        name="gemma3",
        display_name="Gemma 3 4B",
        size_gb=2.5,
        params_billions=4.0,
        requires_gpu=False,
        min_ram_gb=8,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=True,
        context_window=8192
    ),
    
    "llava_med": ModelSpecs(
        name="llava_med",
        display_name="LLaVA-Med 13B",
        size_gb=7.4,
        params_billions=13.0,
        requires_gpu=False,
        min_ram_gb=16,
        min_vram_gb=None,
        quantization="Q4_0",
        provider="local",
        platform="ollama",
        supports_vision=True,
        context_window=4096
    ),
    
    "groq": ModelSpecs(
        name="groq",
        display_name="Groq Llama 3.2 11B",
        size_gb=None,  # Cloud model
        params_billions=11.0,
        requires_gpu=False,
        min_ram_gb=4,
        min_vram_gb=None,
        quantization=None,
        provider="cloud",
        platform="groq",  # ← This makes it show as "Groq"!
        supports_vision=True,
        context_window=8192
    ),
}


# ============================================================================
# PRICING DATABASE (per 1M tokens + per image)
# ============================================================================

MODEL_PRICING = {
    # Vision models
    "gpt-4o": {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "per_image": 0.00765,  # 1024x1024 image
        "currency": "USD"
    },
    "claude-3-5-sonnet-20241022": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "per_image": 0.012,
        "currency": "USD"
    },
    "llama-3.2-11b-vision-preview": {
        "input_per_1m": 0.18,
        "output_per_1m": 0.18,
        "per_image": 0.0,  # Included
        "currency": "USD"
    },
    "gemini-1.5-flash": {
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
        "per_image": 0.0,  # Included
        "currency": "USD"
    },
    
    # LLM models
    "llama-3.3-70b-versatile": {
        "input_per_1m": 0.59,
        "output_per_1m": 0.79,
        "per_image": 0.0,
        "currency": "USD"
    },
    
    # Local models (all free)
    **{
        model: {"input_per_1m": 0.0, "output_per_1m": 0.0, "per_image": 0.0, "currency": "USD"}
        for model in [
            "llava:13b", "llava:latest", "llama3.2-vision", "gemma3:4b", "gemma3:12b",
            "biomedclip", "florence", "llama3.1:8b", "mixtral:8x7b"
        ]
    }
}


# ============================================================================
# HARDWARE REQUIREMENTS TIERS
# ============================================================================

HARDWARE_TIERS = {
    "minimal": {
        "description": "Entry-level laptop",
        "ram_gb": 8,
        "storage_gb": 50,
        "suitable_models": ["llava:latest", "gemma3:4b", "biomedclip"],
        "estimated_cost_usd": 500
    },
    "recommended": {
        "description": "Modern workstation (Mac Mini M4, etc.)",
        "ram_gb": 16,
        "storage_gb": 100,
        "suitable_models": [
            "llava:13b", "llama3.2-vision", "gemma3:12b", "llama3.1:8b"
        ],
        "estimated_cost_usd": 800
    },
    "high_performance": {
        "description": "Server-grade hardware",
        "ram_gb": 32,
        "storage_gb": 500,
        "suitable_models": ["mixtral:8x7b", "all_models"],
        "estimated_cost_usd": 2000
    },
    "cloud": {
        "description": "API-based, any device",
        "ram_gb": 4,
        "storage_gb": 10,
        "suitable_models": [
            "gpt-4o", "claude-3-5-sonnet-20241022", 
            "llama-3.2-11b-vision-preview", "gemini-1.5-flash"
        ],
        "estimated_cost_usd": 300
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_spec(model_name: str) -> Optional[ModelSpecs]:
    """Get specifications for a model"""
    return VISION_MODEL_SPECS.get(model_name) or LLM_MODEL_SPECS.get(model_name)


def get_hardware_tier_for_model(model_name: str) -> str:
    """Determine minimum hardware tier needed"""
    spec = get_model_spec(model_name)
    if not spec:
        return "unknown"
    
    if spec.provider == "cloud":
        return "cloud"
    elif spec.min_ram_gb >= 32:
        return "high_performance"
    elif spec.min_ram_gb >= 16:
        return "recommended"
    else:
        return "minimal"


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    num_images: int = 1
) -> float:
    """Calculate actual cost for a request"""
    pricing = MODEL_PRICING.get(model_name, {})
    
    input_cost = (input_tokens / 1_000_000) * pricing.get("input_per_1m", 0)
    output_cost = (output_tokens / 1_000_000) * pricing.get("output_per_1m", 0)
    image_cost = num_images * pricing.get("per_image", 0)
    
    return input_cost + output_cost + image_cost


def get_deployment_category(model_name: str) -> str:
    """Categorize model for deployment recommendations"""
    spec = get_model_spec(model_name)
    if not spec:
        return "unknown"
    
    if spec.provider == "local":
        if spec.size_gb and spec.size_gb < 5:
            return "ultra_low_cost_local"
        else:
            return "standard_local"
    else:
        # Cloud models
        pricing = MODEL_PRICING.get(model_name, {})
        typical_cost = calculate_cost(model_name, 1000, 500, 1)
        
        if typical_cost < 0.001:
            return "ultra_fast_cloud"
        elif typical_cost < 0.01:
            return "balanced_cloud"
        else:
            return "premium_cloud"