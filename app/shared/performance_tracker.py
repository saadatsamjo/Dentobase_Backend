# app/shared/performance_tracker.py

"""
Performance Tracking System
Collects REAL metrics during vision/LLM tests for thesis analysis
"""
import json
import time
import psutil
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from app.shared.model_specs import get_hardware_tier_for_model, get_model_spec, calculate_cost, MODEL_PRICING


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for ML/Software Engineering thesis.
    
    Focus: System performance, resource efficiency, cost economics.
    NOT clinical details (those belong in medical evaluation).
    """
    # Identification & Core Metrics (Non-Default)
    model_name: str
    test_id: str
    timestamp: str
    inference_time_ms: float
    peak_ram_mb: float

    # ══════════════════════════════════════════════════════════
    # SYSTEM PERFORMANCE METRICS (Core SE/ML) - With Defaults
    # ══════════════════════════════════════════════════════════
    
    # Latency (critical for real-time systems)
    time_to_first_token_ms: Optional[float] = None  # For streaming models
    tokens_per_second: Optional[float] = None
    
    # Throughput (scalability)
    concurrent_requests: int = 1
    queue_wait_time_ms: Optional[float] = None
    
    # ══════════════════════════════════════════════════════════
    # RESOURCE EFFICIENCY (Deployment Planning)
    # ══════════════════════════════════════════════════════════
    
    # Memory usage
    avg_ram_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    
    # CPU/GPU utilization
    avg_cpu_percent: Optional[float] = None
    avg_gpu_percent: Optional[float] = None
    
    # Storage
    model_size_gb: Optional[float] = None
    cache_size_mb: Optional[float] = None
    
    # ══════════════════════════════════════════════════════════
    # COST ECONOMICS (TCO Analysis)
    # ══════════════════════════════════════════════════════════
    
    # Token counts (for cost calculation)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Actual costs
    cost_usd: Optional[float] = None
    cost_per_token: Optional[float] = None
    
    # ══════════════════════════════════════════════════════════
    # QUALITY METRICS (ML Model Performance)
    # ══════════════════════════════════════════════════════════
    
    # Response quality
    response_complete: bool = True  # All required fields present
    schema_valid: bool = True  # Matches expected schema
    citation_count: int = 0  # Number of reference pages cited
    avg_citation_quality: Optional[float] = None  # 0-1 score
    
    # Model confidence (if available)
    confidence_score: Optional[float] = None
    
    # ══════════════════════════════════════════════════════════
    # RELIABILITY METRICS (System Stability)
    # ══════════════════════════════════════════════════════════
    
    # Success/failure tracking
    request_success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # ══════════════════════════════════════════════════════════
    # METADATA (Test Context)
    # ══════════════════════════════════════════════════════════
    
    # Test configuration
    test_type: str = "vision_analysis"  # vision_analysis, rag_query, full_cdss
    model_provider: str = "local"  # local, cloud
    hardware_tier: str = "recommended"  # minimal, recommended, high_performance
    context_provided: Optional[bool] = None
    tooth_number_provided: Optional[bool] = None


class PerformanceTracker:
    """Tracks and persists performance metrics"""
    
    def __init__(self, results_file: str = "evaluation_results.json"):
        self.results_file = Path(results_file)
        self.results: List[PerformanceMetrics] = []
        self._load_existing()
    
    def _load_existing(self):
        """Load existing results from file"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    self.results = [
                        PerformanceMetrics(**item) for item in data
                    ]
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")
                self.results = []
    
    def record(
        self,
        model_name: str,
        inference_time_ms: float,
        # Quality metrics
        response_complete: bool = True,
        schema_valid: bool = True,
        citation_count: int = 0,
        confidence_score: Optional[float] = None,
        # Cost metrics
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        # System metrics
        tokens_per_second: Optional[float] = None,
        # Reliability
        request_success: bool = True,
        error_type: Optional[str] = None,
        # Metadata
        test_type: str = "vision_analysis",
        model_provider: str = "local",
        context_provided: Optional[bool] = None,
        tooth_number_provided: Optional[bool] = None
    ):
        """
        Record performance metrics.
        
        Focus on:
        - System performance ✅
        - Resource efficiency ✅
        - Cost economics ✅
        - Quality/reliability ✅
        """
        # Get system metrics
        process = psutil.Process(os.getpid())
        peak_ram_mb = process.memory_info().rss / 1024 / 1024
        avg_cpu = process.cpu_percent(interval=0.1)
        
        # Calculate derived metrics
        if input_tokens and output_tokens:
            total_tokens = input_tokens + output_tokens
            cost_usd = cost_usd or calculate_cost(model_name, input_tokens, output_tokens)
            cost_per_token = cost_usd / total_tokens if total_tokens > 0 else None
        else:
            total_tokens = None
            cost_per_token = None
        
        if output_tokens and inference_time_ms > 0:
            tokens_per_second = (output_tokens / inference_time_ms) * 1000
        
        # Get model specs
        spec = get_model_spec(model_name)
        model_size_gb = spec.size_gb if spec else None
        hardware_tier = get_hardware_tier_for_model(model_name)
        
        # Create metric
        metric = PerformanceMetrics(
            model_name=model_name,
            test_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            # Performance
            inference_time_ms=inference_time_ms,
            tokens_per_second=tokens_per_second,
            # Resources
            peak_ram_mb=peak_ram_mb,
            avg_cpu_percent=avg_cpu,
            model_size_gb=model_size_gb,
            # Cost
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_per_token=cost_per_token,
            # Quality
            response_complete=response_complete,
            schema_valid=schema_valid,
            citation_count=citation_count,
            confidence_score=confidence_score,
            # Reliability
            request_success=request_success,
            error_type=error_type,
            # Metadata
            test_type=test_type,
            model_provider=model_provider,
            hardware_tier=hardware_tier
        )
        
        self.results.append(metric)
        self._save()
    
    def _save(self):
        """Save results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(
                [asdict(r) for r in self.results],
                f,
                indent=2
            )
    
    def get_stats(self, model_name: str) -> Dict:
        """Get aggregated statistics for a model"""
        model_results = [r for r in self.results if r.model_name == model_name]
        
        if not model_results:
            return {
                "model": model_name,
                "num_tests": 0,
                "status": "no_data"
            }
        
        times = [r.inference_time_ms for r in model_results]
        costs = [r.cost_usd for r in model_results if r.cost_usd is not None]
        confidences = [r.confidence_score for r in model_results if r.confidence_score]
        
        return {
            "model": model_name,
            "num_tests": len(model_results),
            "speed": {
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "avg_sec": sum(times) / len(times) / 1000
            },
            "cost": {
                "avg_usd": sum(costs) / len(costs) if costs else 0.0,
                "total_usd": sum(costs) if costs else 0.0
            },
            "quality": {
                "avg_confidence": sum(confidences) / len(confidences) if confidences else None
            }
        }
    
    def get_all_stats(self) -> List[Dict]:
        """Get stats for all tested models"""
        unique_models = set(r.model_name for r in self.results)
        return [self.get_stats(model) for model in unique_models]
    
    def clear(self):
        """Clear all results (use for fresh evaluation run)"""
        self.results = []
        self._save()
    
    def export_csv(self, filepath: str = "evaluation_results.csv"):
        """Export results to CSV for analysis"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            if not self.results:
                return
            
            writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))
    
    def get_comparison_data(self) -> Dict:
        """
        Get data formatted for thesis comparison tables.
        
        Returns dict with all stats + model specs combined.
        """
        from app.shared.model_specs import get_model_spec, get_hardware_tier_for_model
        
        comparison = []
        
        for model_name in set(r.model_name for r in self.results):
            stats = self.get_stats(model_name)
            spec = get_model_spec(model_name)
            
            if not spec:
                continue
            
            # Calculate typical cost for comparison
            pricing = MODEL_PRICING.get(model_name, {})
            typical_cost = calculate_cost(model_name, 1000, 500, 1)
            
            comparison.append({
                "model": spec.display_name,
                "provider": spec.provider,
                "platform": spec.platform,
                "size_gb": spec.size_gb if spec.size_gb else "Cloud",
                "params_b": spec.params_billions if spec.params_billions else "N/A",
                "min_ram_gb": spec.min_ram_gb,
                "hardware_tier": get_hardware_tier_for_model(model_name),
                "avg_speed_sec": round(stats["speed"]["avg_sec"], 1),
                "cost_per_analysis": typical_cost,
                "cost_per_1000": typical_cost * 1000,
                "annual_cost_50_daily": typical_cost * 50 * 365,
                "avg_confidence": stats["quality"]["avg_confidence"],
                "num_tests": stats["num_tests"]
            })
        
        # Sort by cost (free first, then ascending)
        comparison.sort(key=lambda x: (
            0 if x["cost_per_analysis"] == 0 else 1,
            x["cost_per_analysis"]
        ))
        
        return {
            "models": comparison,
            "summary": {
                "total_models_tested": len(comparison),
                "local_models": len([m for m in comparison if m["provider"] == "local"]),
                "cloud_models": len([m for m in comparison if m["provider"] == "cloud"]),
                "total_tests_run": sum(m["num_tests"] for m in comparison)
            }
        }


# Global tracker instance
performance_tracker = PerformanceTracker()