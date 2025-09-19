"""
LEVEL3 Security Evaluation Framework
By Arkadia - In collaboration with AlephAlpha Company
Developed by Reda

A specialized security evaluation framework for Large Language Models,
focusing on prompt injection, jailbreak resistance, and content safety.
"""

__version__ = "0.1.0"
__author__ = "Reda"
__organization__ = "Arkadia"
__collaborator__ = "AlephAlpha Company"

# Import base classes and data structures
from ._base import SecurityTestCase, EvaluationResult, SecurityEvaluator

# Import main components
from .models import OpenRouterModelRegistry, OllamaModelRegistry
from .metrics import create_metric, list_available_metrics
from .evaluator import BatchSecurityEvaluator
from .datasets import DatasetLoader
from .reporting import ReportGenerator
from .utils import load_config, ensure_directories

__all__ = [
    "SecurityTestCase",
    "EvaluationResult",
    "SecurityEvaluator",
    "OpenRouterModelRegistry",
    "OllamaModelRegistry",
    "create_metric",
    "list_available_metrics",
    "BatchSecurityEvaluator",
    "DatasetLoader",
    "ReportGenerator",
    "load_config",
    "ensure_directories",
    "__version__",
    "__author__",
    "__organization__",
    "__collaborator__",
]
