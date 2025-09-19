"""
Utility functions for LEVEL3 Security Evaluation Framework.
By Arkadia - In collaboration with AlephAlpha Company
Developed by Reda
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and config files."""
    config = {
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "hf_token": os.getenv("HF_TOKEN"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "default_model": os.getenv("LEVEL3_DEFAULT_MODEL", "gpt-4o"),
        "default_metrics": os.getenv("LEVEL3_DEFAULT_METRICS", "jailbreak_sentinel,nemo_guard"),
    }

    return config


def ensure_directories():
    """Ensure all necessary directories exist."""
    dirs = [
        "level3/datasets/promptLib",
        "results",
        "reports",
        "logs"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a value as a percentage string."""
    return f"{value:.{decimals}f}%"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def validate_environment() -> Dict[str, Any]:
    """Validate the environment and return status information."""
    status = {
        "openrouter_available": bool(os.getenv("OPENROUTER_API_KEY")),
        "hf_token_available": bool(os.getenv("HF_TOKEN")),
        "ollama_running": False,
    }

    # Check Ollama (simplified check)
    try:
        import requests
        response = requests.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags", timeout=2)
        status["ollama_running"] = response.status_code == 200
    except:
        status["ollama_running"] = False

    return status


def create_sample_config(output_path: str = ".env.example"):
    """Create a sample configuration file."""
    config_content = """# LEVEL3 Security Evaluation Framework Configuration
# By Arkadia - In collaboration with AlephAlpha Company
# Developed by Reda

# OpenRouter API Key (required for OpenRouter models)
OPENROUTER_API_KEY=your-openrouter-api-key-here

# HuggingFace Token (optional, for enhanced metrics)
HF_TOKEN=your-huggingface-token-here

# Ollama Configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434

# Default Settings
LEVEL3_DEFAULT_MODEL=gpt-4o
LEVEL3_DEFAULT_METRICS=jailbreak_sentinel,nemo_guard
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"Sample configuration created: {output_path}")