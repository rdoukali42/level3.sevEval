"""
Base classes and data structures for LEVEL3 Security Evaluation Framework.
By Arkadia - In collaboration with AlephAlpha Company
Developed by Reda
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class SecurityTestCase:
    """Represents a single security test case for evaluation."""

    id: str
    prompt: str
    expected_safe: bool = True
    category: str = "general"
    metadata: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary for serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected_safe": self.expected_safe,
            "category": self.category,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityTestCase":
        """Create test case from dictionary."""
        data_copy = data.copy()
        if "created_at" in data_copy:
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])
        return cls(**data_copy)


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case with a metric."""

    test_case_id: str
    metric_name: str
    score: float
    is_safe: bool
    details: Dict[str, Any] = None
    raw_output: Any = None
    evaluated_at: datetime = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.evaluated_at is None:
            self.evaluated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "test_case_id": self.test_case_id,
            "metric_name": self.metric_name,
            "score": self.score,
            "is_safe": self.is_safe,
            "details": self.details,
            "raw_output": self.raw_output,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


class BaseModel(ABC):
    """Abstract base class for all model implementations."""

    def __init__(self, name: str, model_id: str = None):
        self.name = name
        self.model_id = model_id or name

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async version of generate."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model_id='{self.model_id}')"

    def __repr__(self) -> str:
        return self.__str__()


class BaseMetric(ABC):
    """Abstract base class for all security metrics."""

    def __init__(self, name: str, description: str = "", threshold: float = 0.5):
        self.name = name
        self.description = description
        self.threshold = threshold
        self._last_score: Optional[float] = None
        self._last_details: Dict[str, Any] = {}

    @abstractmethod
    def evaluate(self, test_case: SecurityTestCase, model_output: str) -> EvaluationResult:
        """Evaluate a test case and model output."""
        pass

    @abstractmethod
    async def aevaluate(self, test_case: SecurityTestCase, model_output: str) -> EvaluationResult:
        """Async version of evaluate."""
        pass

    def is_safe(self, score: float) -> bool:
        """Determine if a score indicates safe behavior."""
        return score >= self.threshold

    @property
    def last_score(self) -> Optional[float]:
        """Get the last evaluation score."""
        return self._last_score

    @property
    def last_details(self) -> Dict[str, Any]:
        """Get the last evaluation details."""
        return self._last_details

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', threshold={self.threshold})"

    def __repr__(self) -> str:
        return self.__str__()


class SecurityEvaluator:
    """Main evaluator class that coordinates models and metrics."""

    def __init__(self, model: BaseModel, metrics: List[BaseMetric]):
        self.model = model
        self.metrics = metrics
        self.results: List[EvaluationResult] = []

    def evaluate_test_case(self, test_case: SecurityTestCase) -> List[EvaluationResult]:
        """Evaluate a single test case with all metrics."""
        # Generate model response
        model_output = self.model.generate(test_case.prompt)

        # Evaluate with all metrics
        results = []
        for metric in self.metrics:
            result = metric.evaluate(test_case, model_output)
            results.append(result)
            self.results.append(result)

        return results

    async def aevaluate_test_case(self, test_case: SecurityTestCase) -> List[EvaluationResult]:
        """Async version of evaluate_test_case."""
        # Generate model response
        model_output = await self.model.agenerate(test_case.prompt)

        # Evaluate with all metrics
        results = []
        for metric in self.metrics:
            result = await metric.aevaluate(test_case, model_output)
            results.append(result)
            self.results.append(result)

        return results

    def evaluate_dataset(self, test_cases: List[SecurityTestCase]) -> Dict[str, Any]:
        """Evaluate an entire dataset."""
        all_results = []

        for test_case in test_cases:
            results = self.evaluate_test_case(test_case)
            all_results.extend(results)

        # Aggregate results
        return self._aggregate_results(all_results)

    async def aevaluate_dataset(self, test_cases: List[SecurityTestCase]) -> Dict[str, Any]:
        """Async version of evaluate_dataset."""
        all_results = []

        for test_case in test_cases:
            results = await self.aevaluate_test_case(test_case)
            all_results.extend(results)

        # Aggregate results
        return self._aggregate_results(all_results)

    def _aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate evaluation results into summary statistics."""
        if not results:
            return {}

        # Group by metric
        metric_results = {}
        for result in results:
            if result.metric_name not in metric_results:
                metric_results[result.metric_name] = []
            metric_results[result.metric_name].append(result)

        # Calculate summary statistics
        summary = {
            "total_evaluations": len(results),
            "model": self.model.name,
            "metrics": {},
            "overall_score": 0.0,
            "safe_percentage": 0.0,
        }

        total_safe = 0
        total_score = 0.0

        for metric_name, metric_results_list in metric_results.items():
            safe_count = sum(1 for r in metric_results_list if r.is_safe)
            avg_score = sum(r.score for r in metric_results_list) / len(metric_results_list)

            summary["metrics"][metric_name] = {
                "evaluations": len(metric_results_list),
                "safe_count": safe_count,
                "unsafe_count": len(metric_results_list) - safe_count,
                "safe_percentage": (safe_count / len(metric_results_list)) * 100,
                "average_score": avg_score,
            }

            total_safe += safe_count
            total_score += avg_score

        summary["overall_score"] = total_score / len(metric_results)
        summary["safe_percentage"] = (total_safe / len(results)) * 100

        return summary