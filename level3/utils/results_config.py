"""
Results configuration utilities for LEVEL3 Security Evaluation Framework.
By Arkadia - In collaboration with AlephAlpha Company
Developed by Reda

Provides automatic file organization for different output formats.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ResultsConfig:
    """Configuration for organizing evaluation results."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize results configuration.
        
        Args:
            base_dir: Base directory for results. Defaults to level3/results/
        """
        if base_dir is None:
            # Find the level3 package directory
            current_file = Path(__file__)
            level3_dir = current_file.parent.parent  # Go up from utils to level3
            base_dir = level3_dir / "results"
        
        self.base_dir = Path(base_dir)
        self.json_dir = self.base_dir / "json"
        self.html_dir = self.base_dir / "html" 
        self.markdown_dir = self.base_dir / "markdown"
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all result directories exist."""
        for dir_path in [self.json_dir, self.html_dir, self.markdown_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(
        self, 
        format: str, 
        filename: Optional[str] = None,
        prefix: str = "evaluation",
        include_timestamp: bool = True
    ) -> Path:
        """Get the appropriate output path for a given format.
        
        Args:
            format: Output format ('json', 'html', 'markdown')
            filename: Custom filename. If None, auto-generates one.
            prefix: Prefix for auto-generated filenames
            include_timestamp: Whether to include timestamp in auto-generated names
            
        Returns:
            Path object for the output file
        """
        format = format.lower()
        
        # Get the appropriate directory
        if format == "json":
            output_dir = self.json_dir
            extension = ".json"
        elif format == "html":
            output_dir = self.html_dir
            extension = ".html"
        elif format in ["markdown", "md"]:
            output_dir = self.markdown_dir
            extension = ".md"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Generate filename if not provided
        if filename is None:
            if include_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prefix}_{timestamp}{extension}"
            else:
                filename = f"{prefix}{extension}"
        else:
            # Ensure correct extension
            filename = Path(filename)
            if filename.suffix.lower() != extension:
                filename = filename.with_suffix(extension)
            filename = str(filename)
        
        return output_dir / filename
    
    def get_json_path(self, filename: Optional[str] = None, **kwargs) -> Path:
        """Get path for JSON output."""
        return self.get_output_path("json", filename, **kwargs)
    
    def get_html_path(self, filename: Optional[str] = None, **kwargs) -> Path:
        """Get path for HTML output."""
        return self.get_output_path("html", filename, **kwargs)
    
    def get_markdown_path(self, filename: Optional[str] = None, **kwargs) -> Path:
        """Get path for Markdown output."""
        return self.get_output_path("markdown", filename, **kwargs)
    
    def resolve_output_path(self, output_path: Optional[str], default_format: str = "json") -> Path:
        """Resolve output path, using organized structure if no path specified.
        
        Args:
            output_path: User-specified output path, or None for auto
            default_format: Default format to use if auto-generating
            
        Returns:
            Resolved Path object
        """
        if output_path is None:
            return self.get_output_path(default_format)
        
        output_path = Path(output_path)
        
        # If it's just a filename (no directory), put it in the appropriate folder
        if output_path.parent == Path("."):
            # Determine format from extension
            ext = output_path.suffix.lower()
            if ext == ".json":
                return self.json_dir / output_path
            elif ext == ".html":
                return self.html_dir / output_path
            elif ext in [".md", ".markdown"]:
                return self.markdown_dir / output_path
            else:
                # Default to specified format
                return self.get_output_path(default_format, str(output_path))
        
        # User specified full path, use as-is but ensure parent dir exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path


# Global results configuration instance
results_config = ResultsConfig()


def get_results_config() -> ResultsConfig:
    """Get the global results configuration."""
    return results_config


def get_json_output_path(filename: Optional[str] = None, **kwargs) -> Path:
    """Convenience function to get JSON output path."""
    return results_config.get_json_path(filename, **kwargs)


def get_html_output_path(filename: Optional[str] = None, **kwargs) -> Path:
    """Convenience function to get HTML output path."""
    return results_config.get_html_path(filename, **kwargs)


def get_markdown_output_path(filename: Optional[str] = None, **kwargs) -> Path:
    """Convenience function to get Markdown output path."""
    return results_config.get_markdown_path(filename, **kwargs)