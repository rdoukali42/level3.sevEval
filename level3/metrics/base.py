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
                import os
                import sys
                
                # Set environment variables before any imports
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"
                os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
                os.environ["NUMEXPR_NUM_THREADS"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                os.environ["HF_HUB_OFFLINE"] = "0"
                
                # Disable multiprocessing entirely
                import multiprocessing
                if hasattr(multiprocessing, 'set_start_method'):
                    try:
                        multiprocessing.set_start_method('spawn', force=True)
                    except RuntimeError:
                        pass  # Already set
                
                from transformers import pipeline, logging, AutoTokenizer, AutoModelForSequenceClassification
                
                # Suppress all warnings
                logging.set_verbosity_error()
                import warnings
                warnings.filterwarnings("ignore")
                
                # Load model and tokenizer separately to avoid pipeline subprocess issues
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    use_fast=False,  # Disable fast tokenizers
                    local_files_only=False
                )
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id,
                    torch_dtype="auto",
                    device_map=None,  # No device mapping
                    local_files_only=False
                )
                
                # Create pipeline with explicit components
                self._pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # Force CPU
                    return_all_scores=True,
                    framework="pt"
                )
                
            except ImportError:
                raise ImportError("transformers package is required for HuggingFace metrics")
            except Exception as e:
                raise RuntimeError(f"Failed to load pipeline for {self.model_id}: {e}")
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