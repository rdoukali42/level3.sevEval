"""
Base metric classes for LEVEL3 Security Evaluation Framework.
By Arkadia - In collaboration with AlephAlpha Company
Developed by Reda
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from level3._base import BaseMetric, SecurityTestCase, EvaluationResult


class HuggingFaceMetric(BaseMetric):
    """Base class for HuggingFace-based security metrics."""

    def __init__(self, name: str, model_id: str, description: str = "", threshold: float = 0.5):
        super().__init__(name, description, threshold)
        self.model_id = model_id
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load the HuggingFace pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-classification" if "sentiment" in self.model_id.lower() else "text-generation",
                    model=self.model_id,
                    device_map="auto",
                    torch_dtype="auto"
                )
            except ImportError:
                raise ImportError("transformers package is required for HuggingFace metrics")
        return self._pipeline

    def _preprocess_input(self, test_case: SecurityTestCase, model_output: str) -> str:
        """Preprocess input for the metric. Override in subclasses."""
        return model_output

    def _postprocess_output(self, raw_output: Any) -> Dict[str, Any]:
        """Postprocess raw model output. Override in subclasses."""
        return {"raw_output": raw_output}


class SafetyMetric(BaseMetric):
    """Base class for safety-focused metrics."""

    def __init__(self, name: str, description: str = "", threshold: float = 0.5):
        super().__init__(name, description, threshold)

    def _calculate_safety_score(self, details: Dict[str, Any]) -> float:
        """Calculate safety score from evaluation details."""
        # Default implementation - override in subclasses
        return 1.0 if details.get("is_safe", True) else 0.0