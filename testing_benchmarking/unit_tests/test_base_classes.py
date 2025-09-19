"""
Unit tests for LEVEL3 Security Evaluation Framework.

Run with: pytest unit_tests/test_base_classes.py -v
"""

import pytest
from level3._base import SecurityTestCase, EvaluationResult, BaseModel, BaseMetric


class TestSecurityTestCase:
    """Test SecurityTestCase data structure."""

    def test_create_test_case(self):
        """Test creating a basic security test case."""
        test_case = SecurityTestCase(
            id="test_001",
            prompt="What is the capital of France?",
            expected_safe=True,
            category="general"
        )

        assert test_case.id == "test_001"
        assert test_case.prompt == "What is the capital of France?"
        assert test_case.expected_safe is True
        assert test_case.category == "general"
        assert test_case.metadata == {}
        assert test_case.created_at is not None

    def test_test_case_serialization(self):
        """Test converting test case to/from dict."""
        original = SecurityTestCase(
            id="test_002",
            prompt="Test prompt",
            expected_safe=False,
            category="jailbreak",
            metadata={"source": "test"}
        )

        # Convert to dict
        data = original.to_dict()

        # Convert back from dict
        restored = SecurityTestCase.from_dict(data)

        assert restored.id == original.id
        assert restored.prompt == original.prompt
        assert restored.expected_safe == original.expected_safe
        assert restored.category == original.category
        assert restored.metadata == original.metadata


class TestEvaluationResult:
    """Test EvaluationResult data structure."""

    def test_create_evaluation_result(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            test_case_id="test_001",
            metric_name="test_metric",
            score=0.8,
            is_safe=True,
            details={"confidence": 0.9},
            model_name="test_model"
        )

        assert result.test_case_id == "test_001"
        assert result.metric_name == "test_metric"
        assert result.score == 0.8
        assert result.is_safe is True
        assert result.details == {"confidence": 0.9}
        assert result.model_name == "test_model"
        assert result.timestamp is not None


class MockMetric(BaseMetric):
    """Mock metric for testing."""

    def evaluate(self, test_case, model_output):
        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name=self.name,
            score=0.5,
            is_safe=True,
            details={},
            model_name="mock_model"
        )

    async def aevaluate(self, test_case, model_output):
        return self.evaluate(test_case, model_output)


class TestBaseMetric:
    """Test BaseMetric abstract class."""

    def test_metric_creation(self):
        """Test creating a metric instance."""
        metric = MockMetric("test_metric", "Test metric description", 0.7)

        assert metric.name == "test_metric"
        assert metric.description == "Test metric description"
        assert metric.threshold == 0.7

    def test_is_safe_method(self):
        """Test the is_safe method."""
        metric = MockMetric("test_metric", threshold=0.5)

        assert metric.is_safe(0.6) is True   # Above threshold
        assert metric.is_safe(0.4) is False  # Below threshold
        assert metric.is_safe(0.5) is True   # At threshold

    def test_string_representation(self):
        """Test string representation of metric."""
        metric = MockMetric("test_metric", threshold=0.5)
        expected = "MockMetric(name='test_metric', threshold=0.5)"
        assert str(metric) == expected
        assert repr(metric) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])