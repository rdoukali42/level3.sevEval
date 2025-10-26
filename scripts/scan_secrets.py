#!/usr/bin/env python3
"""Simple repository secret scanner.

This script searches repository files for common secret patterns and the
specific 'hadcodec' variable. It is intentionally conservative and may
produce false positives; treat findings as signals to review files.

Exit codes:
 - 0: no potential secrets found
 - 1: potential secrets found
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Regex patterns to search for (case-insensitive where useful).
PATTERNS = {
    "PRIVATE_KEY_BLOCK": re.compile(r"-----BEGIN (?:RSA |EC |)PRIVATE KEY-----"),
    "AWS_ACCESS_KEY": re.compile(r"AKIA[0-9A-Z]{16}"),
    "POSSIBLE_API_KEY_ASSIGN": re.compile(r"(?i)(?:api[_-]?key|token|secret|hf_token)\s*[=:]\s*[\"']?([A-Za-z0-9\-_.]{8,})[\"']?"),
    "HADCODEC_VAR": re.compile(r"(?i)hadcodec"),
    "PRIVATE_KEY_PEM": re.compile(r"-----BEGIN.*PRIVATE KEY-----"),
}

SKIP_DIRS = {"venv", ".venv", "env", "node_modules", ".git", "__pycache__"}


def is_text_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
            if b"\0" in chunk:
                return False
    except Exception:
        return False
    return True


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_no, pattern_name, line) matches."""
    matches: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception:
            return matches

    for name, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            # Determine line number
            start = m.start()
            line_no = text.count("\n", 0, start) + 1
            snippet = text.splitlines()[line_no - 1][:200]
            matches.append((line_no, name, snippet))

    # Heuristic: look for long base64-like strings assigned to env-like names
    long_assign = re.compile(r"(?i)(?:key|token|secret)[\w\-]*\s*[=:]\s*[\"']([A-Za-z0-9_\-]{32,})[\"']")
    for m in long_assign.finditer(text):
        start = m.start()
        line_no = text.count("\n", 0, start) + 1
        snippet = text.splitlines()[line_no - 1][:200]
        matches.append((line_no, "LONG_SECRET_ASSIGN", snippet))

    return matches


def main() -> int:
    findings = {}

    for root, dirs, files in os.walk(ROOT):
        # mutate dirs in-place to skip
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            path = Path(root) / fname
            # skip binary
            if not is_text_file(path):
                continue
            # skip some large artifacts
            if path.suffix in {".png", ".jpg", ".jpeg", ".pb", ".pkl", ".h5"}:
                continue

            matches = scan_file(path)
            if matches:
                findings[str(path.relative_to(ROOT))] = matches

    if not findings:
        print("No potential secrets found.")
        return 0

    print("Potential secrets found:")
    for p, hits in findings.items():
        print(f"\nFile: {p}")
        for line_no, name, snippet in hits:
            print(f"  - Line {line_no}: {name}: {snippet}")

    print("\nIf these are false positives (placeholders), verify and ignore as needed. If they are real secrets, remove them and rotate the credentials immediately.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
