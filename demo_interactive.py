#!/usr/bin/env python3
"""
Demo script for LEVEL3 Interactive CLI.

This script demonstrates how to use the interactive graphical CLI
for security evaluation without requiring API keys.

Usage:
    python3 demo_interactive.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def demo_interactive_cli():
    """Demonstrate the interactive CLI features."""

    print("🎮 LEVEL3 Interactive GUI Demo")
    print("=" * 40)

    print("\n📋 New Mouse-Driven Interface:")
    print("1. python3 -m level3.cli interactive")
    print("   - Full graphical interface with mouse support")
    print("   - Click buttons and select from dropdowns")
    print("   - Visual progress bars and status updates")
    print("   - Real-time evaluation feedback")

    print("\n🔧 Built with Textual (TUI Framework):")
    print("• Mouse-clickable buttons and selections")
    print("• Visual progress indicators")
    print("• Multi-screen navigation")
    print("• Error notifications and status messages")
    print("• Keyboard shortcuts (Ctrl+C to quit, Escape to go back)")

    print("\n📖 Interactive GUI Features:")
    print("✓ [Mouse] Click to select model types (OpenRouter/Ollama)")
    print("✓ [Mouse] Click to choose specific models from dropdowns")
    print("✓ [Mouse] Click to add/remove security metrics")
    print("✓ [Mouse] Click to create datasets or select existing files")
    print("✓ [Mouse] Click 'Start Evaluation' to run security tests")
    print("✓ [Mouse] View results with clickable report generation")
    print("✓ [Mouse] Navigate between screens with button clicks")

    print("\n🚀 Quick Start:")
    print("1. Activate venv: source venv/bin/activate")
    print("2. Launch GUI: python3 -m level3.cli interactive")
    print("3. Use mouse to click through the guided interface")
    print("4. View results and generate reports")

    print("\n⌨️  Keyboard Shortcuts:")
    print("• Ctrl+C: Quit application")
    print("• Escape: Go back to previous screen")
    print("• Tab: Navigate between UI elements")
    print("• Enter: Activate selected element")

    print("\n🎯 Example Mouse Workflow:")
    print("1. [Click] 'openrouter' from model type dropdown")
    print("2. [Click] 'Next' button")
    print("3. [Click] 'gpt-4' from model dropdown")
    print("4. [Click] 'Next' button")
    print("5. [Click] 'jailbreak_sentinel' then 'Add Metric'")
    print("6. [Click] 'nemo_guard' then 'Add Metric'")
    print("7. [Click] 'Next' button")
    print("8. [Click] 'Create new sample dataset'")
    print("9. [Click] 'Next' button")
    print("10. [Click] 'Start Evaluation' to run security tests")
    print("11. [Click] 'Generate Report' for HTML output")

    print("\n✨ The interactive GUI makes security evaluation")
    print("   accessible through intuitive mouse-driven interactions!")

if __name__ == "__main__":
    demo_interactive_cli()