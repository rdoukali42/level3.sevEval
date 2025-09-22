"""
Core evaluation engine for LEVEL3 Security Evaluation Framework.
By Arkadia - In collaboration with AlephAlpha Company
Developed by Reda
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from level3._base import SecurityEvaluator, SecurityTestCase, EvaluationResult, BaseModel, BaseMetric
from .datasets import DatasetLoader
from .reporting import ReportGenerator
from .utils.results_config import get_results_config


class BatchSecurityEvaluator:
    """Batch evaluator for processing multiple test cases and models."""

    def __init__(self, models: List[BaseModel], metrics: List[BaseMetric]):
        self.models = models
        self.metrics = metrics
        self.results: Dict[str, List[EvaluationResult]] = {}
        self.dataset_loader = DatasetLoader()
        self.report_generator = ReportGenerator()

    def evaluate_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        use_async: bool = False
    ) -> Dict[str, Any]:
        """Evaluate all models on a dataset."""
        # Load dataset
        test_cases = self.dataset_loader.load_dataset(dataset_path)

        if not test_cases:
            raise ValueError(f"No test cases found in dataset: {dataset_path}")

        print(f"Loaded {len(test_cases)} test cases from {dataset_path}")
        print(f"Evaluating {len(self.models)} models with {len(self.metrics)} metrics...")

        # Evaluate each model
        all_results = {}

        for model in self.models:
            print(f"\nEvaluating model: {model.name}")
            evaluator = SecurityEvaluator(model, self.metrics)

            try:
                # Check if we're already in an event loop
                asyncio.get_running_loop()
                # We're in a running loop, run synchronously
                results = self._evaluate_model_sync(evaluator, test_cases)
            except RuntimeError:
                # No running loop, safe to use async if requested
                if use_async:
                    results = asyncio.run(self._evaluate_model_async(evaluator, test_cases))
                else:
                    results = self._evaluate_model_sync(evaluator, test_cases)

            all_results[model.name] = results
            self.results[model.name] = results

        # Generate summary
        summary = self._generate_summary(all_results, test_cases)

        # Save results if output path provided
        saved_path = None
        if output_path is not None or len(all_results) > 0:  # Always save results
            saved_path = self.save_results(output_path, all_results, summary)
            summary["saved_to"] = saved_path

        return summary

    def _evaluate_model_sync(
        self,
        evaluator: SecurityEvaluator,
        test_cases: List[SecurityTestCase]
    ) -> List[EvaluationResult]:
        """Synchronously evaluate a model on all test cases."""
        results = []

        for test_case in tqdm(test_cases, desc="Evaluating"):
            try:
                case_results = evaluator.evaluate_test_case(test_case)
                results.extend(case_results)
            except Exception as e:
                print(f"Error evaluating test case {test_case.id}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    test_case_id=test_case.id,
                    metric_name="error",
                    score=0.0,
                    is_safe=False,
                    details={"error": str(e)},
                    raw_output=None
                )
                results.append(error_result)

        return results

    async def _evaluate_model_async(
        self,
        evaluator: SecurityEvaluator,
        test_cases: List[SecurityTestCase]
    ) -> List[EvaluationResult]:
        """Asynchronously evaluate a model on all test cases."""
        results = []

        for test_case in tqdm(test_cases, desc="Evaluating (async)"):
            try:
                case_results = await evaluator.aevaluate_test_case(test_case)
                results.extend(case_results)
            except Exception as e:
                print(f"Error evaluating test case {test_case.id}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    test_case_id=test_case.id,
                    metric_name="error",
                    score=0.0,
                    is_safe=False,
                    details={"error": str(e)},
                    raw_output=None
                )
                results.append(error_result)

        return results

    def _generate_summary(
        self,
        all_results: Dict[str, List[EvaluationResult]],
        test_cases: List[SecurityTestCase]
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary of evaluation results."""
        summary = {
            "dataset_info": {
                "total_test_cases": len(test_cases),
                "categories": list(set(tc.category for tc in test_cases)),
            },
            "models_evaluated": list(all_results.keys()),
            "metrics_used": list(set(r.metric_name for results in all_results.values() for r in results)),
            "model_summaries": {},
            "overall_summary": {},
        }

        # Generate per-model summaries
        for model_name, results in all_results.items():
            model_summary = self._generate_model_summary(results, test_cases)
            summary["model_summaries"][model_name] = model_summary

        # Generate overall summary
        summary["overall_summary"] = self._generate_overall_summary(summary["model_summaries"])

        return summary

    def _generate_model_summary(
        self,
        results: List[EvaluationResult],
        test_cases: List[SecurityTestCase]
    ) -> Dict[str, Any]:
        """Generate summary for a single model."""
        if not results:
            return {}

        # Group by metric
        metric_groups = {}
        for result in results:
            if result.metric_name not in metric_groups:
                metric_groups[result.metric_name] = []
            metric_groups[result.metric_name].append(result)

        # Calculate metric summaries
        metric_summaries = {}
        total_safe = 0
        total_evaluations = 0

        for metric_name, metric_results in metric_groups.items():
            safe_count = sum(1 for r in metric_results if r.is_safe)
            avg_score = sum(r.score for r in metric_results) / len(metric_results)

            metric_summaries[metric_name] = {
                "evaluations": len(metric_results),
                "safe_count": safe_count,
                "unsafe_count": len(metric_results) - safe_count,
                "safe_percentage": (safe_count / len(metric_results)) * 100,
                "average_score": avg_score,
            }

            total_safe += safe_count
            total_evaluations += len(metric_results)

        return {
            "total_evaluations": total_evaluations,
            "total_safe": total_safe,
            "overall_safe_percentage": (total_safe / total_evaluations) * 100 if total_evaluations > 0 else 0,
            "metrics": metric_summaries,
        }

    def _generate_overall_summary(self, model_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary across all models."""
        if not model_summaries:
            return {}

        total_safe_percentage = sum(
            summary.get("overall_safe_percentage", 0)
            for summary in model_summaries.values()
        ) / len(model_summaries)

        # Find best and worst performing models
        model_scores = {
            model: summary.get("overall_safe_percentage", 0)
            for model, summary in model_summaries.items()
        }

        best_model = max(model_scores, key=model_scores.get)
        worst_model = min(model_scores, key=model_scores.get)

        return {
            "average_safe_percentage": total_safe_percentage,
            "best_performing_model": best_model,
            "worst_performing_model": worst_model,
            "model_ranking": sorted(model_scores.items(), key=lambda x: x[1], reverse=True),
        }

    def save_results(self, output_path: Optional[str], results: Dict[str, Any], summary: Dict[str, Any]):
        """Save evaluation results to file."""
        # Use results configuration to resolve output path
        results_config = get_results_config()
        resolved_path = results_config.resolve_output_path(output_path, "json")
        
        # Convert EvaluationResult objects to dictionaries for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = [
                result.to_dict() if hasattr(result, 'to_dict') else result 
                for result in model_results
            ]
        
        output_data = {
            "results": serializable_results,
            "summary": summary,
            "metadata": {
                "framework": "LEVEL3 Security Evaluation",
                "by": "Arkadia",
                "collaboration": "AlephAlpha Company",
                "developed_by": "Reda",
            }
        }

        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {resolved_path}")
        return str(resolved_path)

    def generate_report(self, results_path: str, report_path: Optional[str] = None, format: str = "html"):
        """Generate a visual report from evaluation results."""
        return self.report_generator.generate_report(results_path, report_path, format)

    def generate_report(self, results_path: str, report_path: str, format: str = "html"):
        """Generate a visual report from evaluation results."""
        self.report_generator.generate_report(results_path, report_path, format)