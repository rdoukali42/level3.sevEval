"""
Metric registry for LEVEL3 Security Evaluation Framework.
By Arkadia - In collaboration with AlephAlpha Company
Developed by Reda

Provides a centralized registry for all security metrics.
"""

from typing import Dict, Type, Any, Optional
from level3._base import BaseMetric
from .jailbreak_sentinel import JailbreakSentinelMetric
from .nemo_guard import NemoGuardMetric
from .wildguard import WildGuardMetric


class MetricRegistry:
    """Registry for security metrics."""

    def __init__(self):
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._register_builtin_metrics()

    def _register_builtin_metrics(self):
        """Register all built-in metrics."""
        # Real HuggingFace metrics
        self.register("jailbreak_sentinel", JailbreakSentinelMetric)
        self.register("nemo_guard", NemoGuardMetric)
        self.register("wildguard", WildGuardMetric)

    def register(self, name: str, metric_class: Type[BaseMetric]):
        """Register a new metric class."""
        if not issubclass(metric_class, BaseMetric):
            raise TypeError(f"Metric class must inherit from BaseMetric: {metric_class}")
        self._metrics[name] = metric_class

    def create_metric(self, name: str, **kwargs) -> BaseMetric:
        """Create a metric instance."""
        if name not in self._metrics:
            available = ", ".join(self._metrics.keys())
            raise ValueError(f"Unknown metric '{name}'. Available: {available}")

        metric_class = self._metrics[name]
        return metric_class(**kwargs)

    def list_metrics(self) -> Dict[str, str]:
        """List all available metrics with descriptions."""
        return {
            name: cls(description="").description if hasattr(cls, 'description') else name
            for name, cls in self._metrics.items()
        }

    def get_metric_class(self, name: str) -> Type[BaseMetric]:
        """Get the metric class for a given name."""
        if name not in self._metrics:
            available = ", ".join(self._metrics.keys())
            raise ValueError(f"Unknown metric '{name}'. Available: {available}")
        return self._metrics[name]


# Global registry instance
metric_registry = MetricRegistry()


def create_metric(name: str, **kwargs) -> BaseMetric:
    """Convenience function to create a metric."""
    return metric_registry.create_metric(name, **kwargs)


def register_metric(name: str, metric_class: Type[BaseMetric]):
    """Convenience function to register a metric."""
    metric_registry.register(name, metric_class)


def list_available_metrics() -> Dict[str, str]:
    """Convenience function to list available metrics."""
    return metric_registry.list_metrics()