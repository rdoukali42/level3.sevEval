"""
Command-line interface for LEVEL3 Security Evaluation Framework.
By Arkadia - In collaboration with AlephAlpha Company
Developed by Reda
"""

import os
import json
import click
from pathlib import Path
from typing import List, Optional
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.live import Live
from rich.spinner import Spinner
from rich.columns import Columns
from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
import time
import asyncio
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

# Textual imports for mouse-supporting TUI
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Header, Footer, Select, Button, Input, Label, ProgressBar, TextArea, Static
from textual.screen import Screen
from textual import events
from textual.binding import Binding

from .models import OpenRouterModelRegistry, OllamaModelRegistry
from .models.mock_model import MockModelRegistry
from .metrics import list_available_metrics, create_metric
from .evaluator import BatchSecurityEvaluator
from .datasets import DatasetLoader


console = Console()


def print_header():
    """Print the LEVEL3 header."""
    header = Text()
    header.append("LEVEL3 Security Evaluation Framework\n", style="bold cyan")
    header.append("By Arkadia - In collaboration with AlephAlpha Company\n", style="italic blue")
    header.append("Developed by Reda", style="yellow")

    console.print(Panel(header, border_style="green"))


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LEVEL3 Security Evaluation Framework - Evaluate LLM security with specialized metrics."""
    print_header()


@cli.command()
@click.option("--model", "-m", help="Model type (openrouter, ollama)")
@click.option("--model-name", "-n", help="Specific model name")
@click.option("--metrics", "-M", help="Comma-separated list of metrics")
@click.option("--dataset", "-d", help="Path to dataset file/directory")
@click.option("--output", "-o", help="Output file for results")
@click.option("--async-eval", is_flag=True, help="Use async evaluation")
def evaluate(model, model_name, metrics, dataset, output, async_eval):
    """Evaluate model security on a dataset."""
    try:
        # Interactive mode if no parameters provided
        if not all([model, model_name, metrics, dataset]):
            console.print("\n[bold]Interactive Evaluation Mode[/bold]")
            model, model_name, metrics, dataset = _interactive_setup()

        # Parse metrics
        metric_names = [m.strip() for m in metrics.split(",")]
        metric_objects = [create_metric(name) for name in metric_names]

        # Create model
        model_obj = _create_model(model, model_name)

        # Create evaluator
        evaluator = BatchSecurityEvaluator([model_obj], metric_objects)

        # Evaluate
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating security...", total=None)

            results = evaluator.evaluate_dataset(
                dataset_path=dataset,
                output_path=output,
                use_async=async_eval
            )

            progress.update(task, completed=True)

        # Display results
        _display_results(results, model_name, metric_names)

        if "saved_to" in results:
            console.print(f"\n[green]✓[/green] Results saved to: {results['saved_to']}")
        elif output:
            console.print(f"\n[green]✓[/green] Results saved to: {output}")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise click.Abort()


@cli.command()
@click.argument("output_path")
@click.option("--num-samples", "-n", default=10, help="Number of sample test cases")
def create_sample_dataset(output_path, num_samples):
    """Create a sample dataset for testing."""
    try:
        loader = DatasetLoader()
        loader.create_sample_dataset(output_path, num_samples)
        console.print(f"[green]✓[/green] Sample dataset created: {output_path}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()


@cli.command()
@click.argument("results_path")
@click.argument("report_path", required=False)
@click.option("--format", "-f", default="html", type=click.Choice(["html", "json", "markdown"]))
def report(results_path, report_path, format):
    """Generate a visual report from evaluation results."""
    try:
        evaluator = BatchSecurityEvaluator([], [])  # Empty for report generation
        saved_path = evaluator.generate_report(results_path, report_path, format)
        console.print(f"[green]✓[/green] Report generated: {saved_path}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()


@cli.command()
def list_models():
    """List available models."""
    console.print("\n[bold]Available Models:[/bold]")

    # OpenRouter models
    console.print("\n[cyan]OpenRouter Models:[/cyan]")
    or_table = Table(show_header=True)
    or_table.add_column("Model Name", style="cyan")
    or_table.add_column("Description", style="white")

    or_models = OpenRouterModelRegistry.list_models()
    for name, model_id in or_models.items():
        or_table.add_row(name, f"openai/{model_id}")

    console.print(or_table)

    # Ollama models (if running)
    if OllamaModelRegistry.is_ollama_running():
        console.print("\n[cyan]Ollama Models (Local):[/cyan]")
        try:
            loader = DatasetLoader()  # Just to get ollama registry access
            ollama_models = OllamaModelRegistry.create_model("temp").list_available_models()

            if ollama_models:
                ol_table = Table(show_header=True)
                ol_table.add_column("Model Name", style="green")
                for model in ollama_models[:10]:  # Show first 10
                    ol_table.add_row(model)
                if len(ollama_models) > 10:
                    ol_table.add_row(f"... and {len(ollama_models) - 10} more")
                console.print(ol_table)
            else:
                console.print("[yellow]No models found. Run 'ollama pull <model>' to install models.[/yellow]")
        except:
            console.print("[yellow]Could not connect to Ollama.[/yellow]")
    else:
        console.print("[yellow]Ollama not running. Start Ollama to use local models.[/yellow]")


@cli.command()
def list_metrics():
    """List available security metrics."""
    console.print("\n[bold]Available Security Metrics:[/bold]")

    metrics_table = Table(show_header=True)
    metrics_table.add_column("Metric Name", style="cyan", no_wrap=True)
    metrics_table.add_column("Description", style="white")

    available_metrics = list_available_metrics()
    for name, description in available_metrics.items():
        metrics_table.add_row(name, description)

    console.print(metrics_table)


@cli.command()
@click.argument("dataset_path")
def validate_dataset(dataset_path):
    """Validate a dataset file."""
    try:
        loader = DatasetLoader()
        test_cases = loader.load_dataset(dataset_path)
        validation = loader.validate_dataset(test_cases)

        if validation["valid"]:
            console.print("[green]✓ Dataset is valid![/green]")

            # Show statistics
            stats_table = Table(show_header=True, title="Dataset Statistics")
            stats_table.add_column("Property", style="cyan")
            stats_table.add_column("Value", style="white")

            stats_table.add_row("Total Cases", str(validation["total_cases"]))
            stats_table.add_row("Safe Cases", str(validation["safe_cases"]))
            stats_table.add_row("Unsafe Cases", str(validation["unsafe_cases"]))

            console.print(stats_table)

            # Show categories
            if validation["categories"]:
                cat_table = Table(show_header=True, title="Categories")
                cat_table.add_column("Category", style="cyan")
                cat_table.add_column("Count", style="white")

                for cat, count in validation["categories"].items():
                    cat_table.add_row(cat, str(count))

                console.print(cat_table)

        else:
            console.print("[red]✗ Dataset validation failed:[/red]")
            for error in validation["errors"]:
                console.print(f"  - {error}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()


def _interactive_setup():
    """Interactive setup for evaluation parameters."""
    # Model selection
    console.print("\n[bold]Model Selection:[/bold]")
    model_type = Prompt.ask("Choose model type", choices=["openrouter", "ollama"], default="openrouter")

    if model_type == "openrouter":
        console.print("Available OpenRouter models: gpt-4o, gpt-4-turbo, claude-3-opus, etc.")
        model_name = Prompt.ask("Enter model name", default="gpt-4o")
    else:
        if not OllamaModelRegistry.is_ollama_running():
            console.print("[red]Ollama is not running. Please start Ollama first.[/red]")
            raise click.Abort()

        console.print("Checking available Ollama models...")
        # This is a simplified check - in real implementation you'd list models
        model_name = Prompt.ask("Enter Ollama model name", default="llama3.1")

    # Metrics selection
    console.print("\n[bold]Metrics Selection:[/bold]")
    available_metrics = list(list_available_metrics().keys())
    console.print(f"Available metrics: {', '.join(available_metrics)}")

    metrics_input = Prompt.ask("Enter metrics (comma-separated)", default="jailbreak_sentinel,nemo_guard")
    metrics = metrics_input

    # Dataset selection
    console.print("\n[bold]Dataset Selection:[/bold]")
    default_dataset = "level3/datasets/promptlib"
    if Path(default_dataset).exists():
        dataset = Prompt.ask("Enter dataset path", default=default_dataset)
    else:
        dataset = Prompt.ask("Enter dataset path")

    return model_type, model_name, metrics, dataset


def _create_model(model_type: str, model_name: str):
    """Create a model instance."""
    if model_type == "openrouter":
        return OpenRouterModelRegistry.create_model(model_name)
    elif model_type == "ollama":
        return OllamaModelRegistry.create_model(model_name)
    elif model_type == "mock":
        return MockModelRegistry.create_model(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _display_results(results: dict, model_name: str, metric_names: List[str]):
    """Display evaluation results in a nice format."""
    summary = results.get("overall_summary", {})
    model_summary = results.get("model_summaries", {}).get(model_name, {})

    # Main results panel
    result_text = Text()
    result_text.append("Security Assessment Results\n\n", style="bold cyan")
    result_text.append(f"Model: {model_name}\n", style="white")
    result_text.append(f"Dataset: {results.get('dataset_info', {}).get('total_test_cases', 0)} test cases\n", style="white")
    result_text.append(f"Metrics: {', '.join(metric_names)}\n\n", style="white")

    safe_pct = model_summary.get("overall_safe_percentage", 0)
    if safe_pct >= 80:
        color = "green"
    elif safe_pct >= 60:
        color = "yellow"
    else:
        color = "red"

    result_text.append(f"{safe_pct:.1f}%", style=f"bold {color}")

    console.print(Panel(result_text, border_style="blue"))

    # Metrics breakdown
    if "metrics" in model_summary:
        metrics_table = Table(show_header=True, title="Metrics Breakdown")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Safe %", style="green", justify="right")
        metrics_table.add_column("Avg Score", style="yellow", justify="right")
        metrics_table.add_column("Evaluations", style="white", justify="right")

        for metric_name, metric_data in model_summary["metrics"].items():
            safe_pct = f"{metric_data['safe_percentage']:.1f}%"
            avg_score = f"{metric_data['average_score']:.3f}"
            evaluations = str(metric_data['evaluations'])

            metrics_table.add_row(metric_name, safe_pct, avg_score, evaluations)

        console.print(metrics_table)


@cli.command()
def interactive():
    """Interactive graphical CLI for security evaluation with mouse support."""
    app = LEVEL3InteractiveApp()
    app.run()


class LEVEL3InteractiveApp(App):
    """Textual-based interactive CLI for LEVEL3 Security Evaluation."""

    CSS_PATH = Path(__file__).parent / "ui_styles.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self):
        super().__init__()
        self.selected_model_type = None
        self.selected_model = None
        self.selected_metrics = []
        self.dataset_path = None
        self.evaluation_results = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())

    def action_quit(self) -> None:
        self.exit()

    def action_back(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()


class WelcomeScreen(Screen):
    """Welcome screen with model type selection."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🎮 LEVEL3 Interactive Security Evaluation", classes="title")
            yield Static("Welcome to the mouse-driven interface for LLM security testing!", classes="description")

            yield Static("📡 Step 1: Choose Model Type", classes="title")
            with Container():
                yield Button("OpenRouter (Cloud-hosted models)", id="openrouter_button", variant="primary")
                yield Button("Ollama (Local models)", id="ollama_button", variant="default")
                yield Button("Mock (Testing models - no API/subprocess)", id="mock_button", variant="success")

            with Horizontal():
                yield Button("Quit", id="quit_button", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "openrouter_button":
            self.app.selected_model_type = "openrouter"
            self.app.push_screen(ModelSelectionScreen())
        elif event.button.id == "ollama_button":
            self.app.selected_model_type = "ollama"
            self.app.push_screen(ModelSelectionScreen())
        elif event.button.id == "mock_button":
            self.app.selected_model_type = "mock"
            self.app.push_screen(ModelSelectionScreen())
        elif event.button.id == "quit_button":
            self.app.exit()


class ModelSelectionScreen(Screen):
    """Screen for selecting specific model."""

    def compose(self) -> ComposeResult:
        model_type = self.app.selected_model_type

        with Vertical():
            yield Static(f"🤖 Step 2: Choose {model_type.title()} Model", classes="title")

            models = self._get_available_models()
            if not models:
                yield Static(f"❌ No {model_type} models available. Please check your configuration.", classes="description")
                yield Button("Back", id="back_button")
                return

            with Container():
                # Display available models as buttons
                with Vertical():
                    for name, model_id in models.items():
                        yield Button(f"{name} - {model_id}", id=f"model_{name}", classes="model-button")

            with Horizontal():
                yield Button("Back", id="back_button")

    def _get_available_models(self):
        """Get available models for the current model type."""
        model_type = self.app.selected_model_type
        if model_type == "openrouter":
            registry = OpenRouterModelRegistry()
            return registry.list_models()
        elif model_type == "ollama":
            try:
                temp_model = OllamaModelRegistry.create_model("temp")
                models_list = temp_model.list_available_models()
                return {model["name"]: f"{model.get('size', 'Unknown size')}" for model in models_list}
            except Exception:
                return {}
        elif model_type == "mock":
            return MockModelRegistry.list_models()
        else:
            return {}

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "back_button":
            self.app.pop_screen()
        elif button_id.startswith("model_"):
            model_name = button_id[6:]  # Remove "model_" prefix
            models = self._get_available_models()
            if model_name in models:
                self.app.selected_model = model_name
                self.app.push_screen(MetricSelectionScreen())
            else:
                self.notify("Invalid model selection!", severity="error")


class MetricSelectionScreen(Screen):
    """Screen for selecting security metrics."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🛡️ Step 3: Choose Security Metrics", classes="title")
            yield Static("Select one or more metrics to evaluate:", classes="description")

            available_metrics = list_available_metrics()

            with ScrollableContainer():
                # Display available metrics as buttons
                with Vertical():
                    for name, description in available_metrics.items():
                        yield Button(f"{name} - {description}", id=f"metric_{name}", classes="metric-button")

            # Display selected metrics
            yield Static("Selected metrics:", id="selected_label")
            yield TextArea("", id="selected_metrics", disabled=True, classes="results-table")

            with Horizontal():
                yield Button("Clear All", id="clear_button", variant="warning")

            with Horizontal():
                yield Button("Back", id="back_button")
                yield Button("Next", id="next_button", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        
        if button_id == "clear_button":
            if not hasattr(self.app, 'selected_metrics'):
                self.app.selected_metrics = []
            self.app.selected_metrics = []
            self._update_selected_display()
            self.notify("Cleared all selected metrics", severity="information")
        elif button_id == "back_button":
            self.app.pop_screen()
        elif button_id == "next_button":
            if not hasattr(self.app, 'selected_metrics') or not self.app.selected_metrics:
                self.notify("Please select at least one metric!", severity="error")
                return
            self.app.push_screen(DatasetSelectionScreen())
        elif button_id.startswith("metric_"):
            metric_name = button_id[7:]  # Remove "metric_" prefix
            if not hasattr(self.app, 'selected_metrics'):
                self.app.selected_metrics = []
            
            if metric_name not in self.app.selected_metrics:
                self.app.selected_metrics.append(metric_name)
                self._update_selected_display()
                self.notify(f"Added {metric_name} to selection", severity="information")
            else:
                self.notify(f"{metric_name} already selected", severity="warning")

    def _update_selected_display(self):
        """Update the display of selected metrics."""
        text_area = self.query_one("#selected_metrics", TextArea)
        if hasattr(self.app, 'selected_metrics') and self.app.selected_metrics:
            text_area.text = "\n".join([f"• {metric}" for metric in self.app.selected_metrics])
        else:
            text_area.text = "No metrics selected yet"


class DatasetSelectionScreen(Screen):
    """Screen for dataset selection and creation."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📁 Step 4: Choose Dataset", classes="title")

            # Check for existing datasets in promptlib folder
            promptlib_dir = Path("level3/datasets/promptlib")
            json_files = []
            
            if promptlib_dir.exists():
                json_files = list(promptlib_dir.glob("*.json"))

            with Container():
                # Display dataset options as buttons
                with Vertical():
                    yield Button("Create new sample dataset", id="dataset_create_new", classes="dataset-button")
                    if json_files:
                        for f in json_files:
                            file_size = f.stat().st_size
                            # Create valid ID by replacing dots and other invalid chars with underscores
                            safe_id = f"dataset_{f.stem.replace('.', '_').replace('-', '_')}"
                            yield Button(f"{f.name} ({file_size} bytes)", id=safe_id, classes="dataset-button")
                    else:
                        yield Static("No datasets found in level3/datasets/promptlib/", classes="description")

            # Sample count input (only shown when creating new)
            yield Static("Number of sample test cases:", id="sample_count_label")
            yield Input(placeholder="5", id="sample_count_input", type="integer")

            with Horizontal():
                yield Button("Back", id="back_button")
                yield Button("Next", id="next_button", variant="primary")

    def on_mount(self) -> None:
        # Initialize with sample count hidden since create_new isn't selected by default
        self._hide_sample_count_ui()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        
        if button_id == "back_button":
            self.app.pop_screen()
        elif button_id == "next_button":
            if not hasattr(self.app, 'selected_dataset_type'):
                self.notify("Please select a dataset option!", severity="error")
                return

            if self.app.selected_dataset_type == "create_new":
                count_input = self.query_one("#sample_count_input", Input)
                try:
                    sample_count = int(count_input.value) if count_input.value else 5
                    self.app.sample_count = sample_count
                except ValueError:
                    self.notify("Please enter a valid number for sample count!", severity="error")
                    return
            
            self.app.push_screen(EvaluationScreen())
        elif button_id == "dataset_create_new":
            self.app.selected_dataset_type = "create_new" 
            self.app.dataset_path = "level3/datasets/promptlib/sample_dataset.json"
            self._show_sample_count_ui()
            self.notify("Selected: Create new sample dataset", severity="information")
        elif button_id.startswith("dataset_") and not button_id == "dataset_create_new":
            # Extract filename from safe ID - need to map back to actual file
            promptlib_dir = Path("level3/datasets/promptlib")
            json_files = list(promptlib_dir.glob("*.json"))
            
            # Find the matching file based on the button ID
            selected_file = None
            for f in json_files:
                safe_id = f"dataset_{f.stem.replace('.', '_').replace('-', '_')}"
                if safe_id == button_id:
                    selected_file = f
                    break
            
            if selected_file:
                self.app.selected_dataset_type = "existing"
                self.app.dataset_path = str(selected_file)
                self._hide_sample_count_ui()
                self.notify(f"Selected dataset: {selected_file.name}", severity="information")
            else:
                self.notify("Could not find selected dataset!", severity="error")

    def _show_sample_count_ui(self):
        """Show sample count input for new dataset creation."""
        count_label = self.query_one("#sample_count_label", Static)
        count_input = self.query_one("#sample_count_input", Input)
        count_label.display = True
        count_input.display = True

    def _hide_sample_count_ui(self):
        """Hide sample count input when using existing dataset."""
        count_label = self.query_one("#sample_count_label", Static)
        count_input = self.query_one("#sample_count_input", Input)
        count_label.display = False
        count_input.display = False

    def _create_sample_dataset(self, path: str, num_samples: int):
        """Create a sample dataset."""
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        test_cases = []
        for i in range(num_samples):
            test_cases.append({
                "id": f"{i:04d}",
                "prompt": f"Sample security test prompt {i+1} for evaluation",
                "expected_safe": True,
                "category": "interactive_sample"
            })

        with open(path, 'w') as f:
            json.dump(test_cases, f, indent=2)


class EvaluationScreen(Screen):
    """Screen showing evaluation progress and results."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🚀 Step 5: Security Evaluation", classes="title")

            # Show configuration
            config_text = f"""
Model Type: {self.app.selected_model_type}
Model: {self.app.selected_model}
Metrics: {', '.join(self.app.selected_metrics)}
Dataset: {self.app.dataset_path}
            """.strip()

            yield Static("Configuration:", classes="description")
            yield TextArea(config_text, disabled=True, id="config_display")

            yield Static("Progress:", classes="description")
            yield ProgressBar(id="progress_bar", total=100)

            yield Static("", id="status_text")

            with Horizontal():
                yield Button("Start Evaluation", id="start_button", variant="primary")
                yield Button("Back", id="back_button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_button":
            self.app.pop_screen()

        elif event.button.id == "start_button":
            self._run_evaluation()

    def _run_evaluation(self):
        """Run the security evaluation."""
        try:
            # Update UI
            progress_bar = self.query_one("#progress_bar", ProgressBar)
            status_text = self.query_one("#status_text", Static)
            start_button = self.query_one("#start_button", Button)
            start_button.disabled = True

            status_text.update("🔄 Initializing evaluation...")
            progress_bar.update(progress=10)

            # Initialize components
            if self.app.selected_model_type == "openrouter":
                registry = OpenRouterModelRegistry()
                model = registry.create_model(self.app.selected_model)
            elif self.app.selected_model_type == "ollama":
                registry = OllamaModelRegistry()
                model = registry.create_model(self.app.selected_model)
            elif self.app.selected_model_type == "mock":
                registry = MockModelRegistry()
                model = registry.create_model(self.app.selected_model)
            else:
                raise ValueError(f"Unknown model type: {self.app.selected_model_type}")

            metrics = [create_metric(name) for name in self.app.selected_metrics]

            status_text.update("📁 Loading dataset...")
            progress_bar.update(progress=30)

            # Handle dataset creation if needed
            if self.app.selected_dataset_type == "create_new":
                dataset_path = "level3/datasets/promptlib/sample_dataset.json"
                self._create_sample_dataset(dataset_path, getattr(self.app, 'sample_count', 5))
                self.app.dataset_path = dataset_path

            dataset_loader = DatasetLoader()
            test_cases = dataset_loader.load_dataset(self.app.dataset_path)

            status_text.update("⚡ Running evaluation...")
            progress_bar.update(progress=50)

            # Run evaluation
            evaluator = BatchSecurityEvaluator([model], metrics)
            results_summary = evaluator.evaluate_dataset(
                self.app.dataset_path,
                output_path=None,
                use_async=True
            )

            # Collect results
            results = []
            for model_results in evaluator.results.values():
                results.extend(model_results)

            progress_bar.update(progress=90)
            status_text.update("📊 Processing results...")

            # Store results
            self.app.evaluation_results = results

            progress_bar.update(progress=100)
            status_text.update("✅ Evaluation complete!")

            # Auto-advance to results screen
            self.app.push_screen(ResultsScreen())

        except Exception as e:
            self.notify(f"Evaluation failed: {str(e)}", severity="error")
            start_button = self.query_one("#start_button", Button)
            start_button.disabled = False


class ResultsScreen(Screen):
    """Screen displaying evaluation results."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📊 Evaluation Results", classes="title")

            if not self.app.evaluation_results:
                yield Static("❌ No results available", classes="description")
                yield Button("Back", id="back_button")
                return

            results = self.app.evaluation_results

            # Summary statistics
            total_evaluations = len(results)
            safe_evaluations = sum(1 for r in results if r.is_safe)
            safe_percentage = (safe_evaluations / total_evaluations * 100) if total_evaluations > 0 else 0

            summary_text = f"""
📈 Overall Summary:
• Total Evaluations: {total_evaluations}
• Safe Evaluations: {safe_evaluations}
• Safety Rate: {safe_percentage:.1f}%

🔍 Configuration:
• Model: {self.app.selected_model_type}/{self.app.selected_model}
• Metrics: {', '.join(self.app.selected_metrics)}
• Dataset: {self.app.dataset_path}
            """.strip()

            yield TextArea(summary_text, disabled=True, id="summary_display")

            # Detailed results table would go here
            yield Static("Detailed results saved to file.", classes="description")

            with Horizontal():
                yield Button("Generate Report", id="report_button", variant="success")
                yield Button("New Evaluation", id="new_eval_button", variant="primary")
                yield Button("Quit", id="quit_button", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "report_button":
            self._generate_report()

        elif event.button.id == "new_eval_button":
            # Reset app state and go back to welcome
            self.app.selected_model_type = None
            self.app.selected_model = None
            self.app.selected_metrics = []
            self.app.dataset_path = None
            self.app.evaluation_results = None
            self.app.pop_screen()  # Remove results screen
            self.app.pop_screen()  # Remove evaluation screen
            self.app.pop_screen()  # Remove dataset screen
            self.app.pop_screen()  # Remove metric screen
            self.app.pop_screen()  # Remove model screen
            # Back to welcome screen

        elif event.button.id == "quit_button":
            self.app.exit()

    def _generate_report(self):
        """Generate HTML report."""
        try:
            from .reporting.generator import ReportGenerator
            from .utils.results_config import get_results_config
            
            with self.app.screen.status("Generating HTML report..."):
                results_config = get_results_config()
                output_file = results_config.get_html_path(prefix="interactive_report")
                
                reporter = ReportGenerator()
                reporter.generate_html_report(self.app.evaluation_results, str(output_file))

            self.notify(f"Report generated: {output_file}", severity="information")
        except Exception as e:
            self.notify(f"Report generation failed: {str(e)}", severity="error")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
