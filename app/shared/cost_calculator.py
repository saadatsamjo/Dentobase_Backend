# app/shared/cost_calculator.py
"""
Cost Calculator for Vision and LLM Models
Provides pricing estimates and cost comparison across all providers
Updated: February 2026
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# PRICING DATA (per 1M tokens, USD)
# ============================================================================

@dataclass
class ModelPricing:
    """Pricing for a single model"""
    input_price: float   # Per 1M input tokens
    output_price: float  # Per 1M output tokens
    notes: str = ""


# Vision Model Pricing (estimates for dental X-ray analysis)
# Note: Vision models typically charge per image + tokens
VISION_MODEL_PRICING = {
    # Premium Cloud (Proprietary)
    "gpt-4o": ModelPricing(2.50, 10.00, "Best accuracy, expensive"),
    "gpt-4-vision-preview": ModelPricing(10.00, 30.00, "Legacy model"),
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, "Excellent reasoning"),
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, "Highest capability"),
    
    # Fast Cloud (New additions)
    "llama-3.2-90b-vision-preview": ModelPricing(0.59, 0.79, "Groq - ultra-fast"),
    "llama-3.2-11b-vision-preview": ModelPricing(0.18, 0.18, "Groq - faster, cheaper"),
    "gemini-2.0-flash-exp": ModelPricing(0.075, 0.30, "Google - best value"),
    "gemini-1.5-pro": ModelPricing(1.25, 5.00, "Google - high capability"),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30, "Google - fast"),
    
    # Local Models (Free)
    "llava:13b": ModelPricing(0.00, 0.00, "Local Ollama - best open-source"),
    "llama3.2-vision": ModelPricing(0.00, 0.00, "Local Ollama - Meta latest"),
    "gemma3:12b": ModelPricing(0.00, 0.00, "Local Ollama - Google multimodal"),
    "gemma3:4b": ModelPricing(0.00, 0.00, "Local Ollama - lightweight"),
    "biomedclip": ModelPricing(0.00, 0.00, "Local - pathology classifier"),
}

# LLM Pricing (text-only, for RAG recommendations)
LLM_MODEL_PRICING = {
    # Cloud - Premium
    "gpt-4o": ModelPricing(2.50, 10.00, "OpenAI latest"),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00, "OpenAI legacy"),
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, "Anthropic best"),
    
    # Cloud - Fast/Cheap (New)
    "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79, "Groq - 70B fast"),
    "mixtral-8x7b-32768": ModelPricing(0.24, 0.24, "Groq - MoE model"),
    "gemini-2.0-flash-exp": ModelPricing(0.075, 0.30, "Google - best value"),
    "gemini-1.5-pro": ModelPricing(1.25, 5.00, "Google - 2M context"),
    
    # Local - Free
    "llama3.1:8b": ModelPricing(0.00, 0.00, "Local Ollama"),
    "mixtral:8x7b": ModelPricing(0.00, 0.00, "Local Ollama"),
    "gemma3:4b": ModelPricing(0.00, 0.00, "Local Ollama"),
}


# ============================================================================
# COST ESTIMATION FUNCTIONS
# ============================================================================

def estimate_vision_cost(
    model: str, 
    num_images: int = 1,
    avg_input_tokens: int = 1000,
    avg_output_tokens: int = 500
) -> Dict[str, float]:
    """
    Estimate cost for vision model inference.
    
    Args:
        model: Model identifier
        num_images: Number of images to analyze
        avg_input_tokens: Average input tokens (prompt + image)
        avg_output_tokens: Average output tokens
    
    Returns:
        Dict with cost breakdown
    """
    pricing = VISION_MODEL_PRICING.get(model)
    if not pricing:
        return {"error": f"Unknown model: {model}"}
    
    # Calculate per-request cost
    input_cost = (avg_input_tokens / 1_000_000) * pricing.input_price
    output_cost = (avg_output_tokens / 1_000_000) * pricing.output_price
    per_image_cost = input_cost + output_cost
    
    return {
        "model": model,
        "cost_per_image": round(per_image_cost, 6),
        "total_cost": round(per_image_cost * num_images, 4),
        "cost_per_1000_images": round(per_image_cost * 1000, 2),
        "pricing_notes": pricing.notes,
    }


def estimate_llm_cost(
    model: str,
    num_requests: int = 1,
    avg_input_tokens: int = 2000,
    avg_output_tokens: int = 500
) -> Dict[str, float]:
    """
    Estimate cost for LLM text generation.
    
    Args:
        model: Model identifier
        num_requests: Number of requests
        avg_input_tokens: Average input tokens (context + prompt)
        avg_output_tokens: Average output tokens
    
    Returns:
        Dict with cost breakdown
    """
    pricing = LLM_MODEL_PRICING.get(model)
    if not pricing:
        return {"error": f"Unknown model: {model}"}
    
    input_cost = (avg_input_tokens / 1_000_000) * pricing.input_price
    output_cost = (avg_output_tokens / 1_000_000) * pricing.output_price
    per_request_cost = input_cost + output_cost
    
    return {
        "model": model,
        "cost_per_request": round(per_request_cost, 6),
        "total_cost": round(per_request_cost * num_requests, 4),
        "cost_per_1000_requests": round(per_request_cost * 1000, 2),
        "pricing_notes": pricing.notes,
    }


def compare_vision_models(
    num_images: int = 1000,
    avg_input_tokens: int = 1000,
    avg_output_tokens: int = 500
) -> List[Dict]:
    """
    Compare costs across all vision models.
    
    Returns:
        List of dicts sorted by cost (cheapest first)
    """
    comparisons = []
    
    for model, pricing in VISION_MODEL_PRICING.items():
        cost_data = estimate_vision_cost(
            model, num_images, avg_input_tokens, avg_output_tokens
        )
        comparisons.append(cost_data)
    
    # Sort by total cost
    return sorted(comparisons, key=lambda x: x.get("total_cost", float('inf')))


def compare_llm_models(
    num_requests: int = 1000,
    avg_input_tokens: int = 2000,
    avg_output_tokens: int = 500
) -> List[Dict]:
    """
    Compare costs across all LLM models.
    
    Returns:
        List of dicts sorted by cost (cheapest first)
    """
    comparisons = []
    
    for model, pricing in LLM_MODEL_PRICING.items():
        cost_data = estimate_llm_cost(
            model, num_requests, avg_input_tokens, avg_output_tokens
        )
        comparisons.append(cost_data)
    
    return sorted(comparisons, key=lambda x: x.get("total_cost", float('inf')))


def estimate_cdss_pipeline_cost(
    vision_model: str,
    llm_model: str,
    num_cases: int = 100
) -> Dict:
    """
    Estimate total cost for complete CDSS pipeline (vision + RAG + LLM).
    
    Assumptions:
    - Vision: 1000 input tokens (image + prompt), 500 output tokens
    - LLM: 2000 input tokens (context + retrieved docs), 500 output tokens
    
    Returns:
        Dict with cost breakdown
    """
    vision_cost_data = estimate_vision_cost(vision_model, num_cases, 1000, 500)
    llm_cost_data = estimate_llm_cost(llm_model, num_cases, 2000, 500)
    
    total_cost = vision_cost_data.get("total_cost", 0) + llm_cost_data.get("total_cost", 0)
    
    return {
        "pipeline": f"{vision_model} + {llm_model}",
        "num_cases": num_cases,
        "vision_cost": vision_cost_data.get("total_cost", 0),
        "llm_cost": llm_cost_data.get("total_cost", 0),
        "total_cost": round(total_cost, 4),
        "cost_per_case": round(total_cost / num_cases, 6) if num_cases > 0 else 0,
        "annual_cost_50_daily": round((total_cost / num_cases) * 50 * 365, 2) if num_cases > 0 else 0,
    }


def get_cheapest_options() -> Dict[str, str]:
    """
    Get the cheapest model for each category.
    
    Returns:
        Dict mapping category to cheapest model
    """
    # Find cheapest non-zero cost models
    vision_costs = [
        (model, pricing.input_price + pricing.output_price)
        for model, pricing in VISION_MODEL_PRICING.items()
        if pricing.input_price > 0 or pricing.output_price > 0
    ]
    
    llm_costs = [
        (model, pricing.input_price + pricing.output_price)
        for model, pricing in LLM_MODEL_PRICING.items()
        if pricing.input_price > 0 or pricing.output_price > 0
    ]
    
    cheapest_vision_cloud = min(vision_costs, key=lambda x: x[1])[0] if vision_costs else None
    cheapest_llm_cloud = min(llm_costs, key=lambda x: x[1])[0] if llm_costs else None
    
    return {
        "cheapest_vision_cloud": cheapest_vision_cloud,
        "cheapest_llm_cloud": cheapest_llm_cloud,
        "cheapest_vision_local": "gemma3:4b",  # Smallest local multimodal
        "cheapest_llm_local": "gemma3:4b",     # Smallest local LLM
        "best_value_vision": "gemini-2.0-flash-exp",  # Best accuracy/cost ratio
        "best_value_llm": "gemini-2.0-flash-exp",
        "fastest_vision": "llama-3.2-11b-vision-preview",  # Groq 11B
        "fastest_llm": "llama-3.3-70b-versatile",  # Groq 70B
    }