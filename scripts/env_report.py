"""
Environment report — prints and saves a full snapshot of the runtime environment.

Captures Python version, OS, git hash, pip freeze, env vars, and library imports.
Run this first after setup to verify everything is installed correctly.

Usage: python scripts/env_report.py
Output: logs/env_report.json
"""

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_git_info() -> dict:
    """Get git commit hash, dirty status, and branch name."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {
            "commit_hash": commit,
            "is_dirty": len(dirty) > 0,
            "branch": branch,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit_hash": "unknown", "is_dirty": "unknown", "branch": "unknown"}


def get_pip_freeze() -> list[str]:
    """Get installed packages via pip freeze."""
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return [line.strip() for line in output.strip().split("\n") if line.strip()]
    except subprocess.CalledProcessError:
        return ["ERROR: pip freeze failed"]


def check_imports() -> dict[str, str]:
    """Try importing each required library, report version or error."""
    libraries = [
        "openai",
        "tiktoken",
        "tenacity",
        "pydantic",
        "orjson",
        "numpy",
        "pandas",
        "sklearn",
        "faiss",
        "rank_bm25",
        "typer",
        "rich",
        "loguru",
        "matplotlib",
        "yaml",
        "tqdm",
        "Bio",
    ]
    results = {}
    for lib in libraries:
        try:
            mod = importlib.import_module(lib)
            results[lib] = getattr(mod, "__version__", "imported (no __version__)")
        except ImportError as e:
            results[lib] = f"MISSING: {e}"
    return results


def check_env_vars() -> dict[str, str]:
    """Check required environment variables."""
    results = {}
    for var in [
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "OPENAI_API_KEY",
        "NCBI_EMAIL",
    ]:
        value = os.environ.get(var)
        if var == "OPENAI_API_KEY" and value:
            # Mask API key — show only first 8 chars
            results[var] = value[:8] + "..." if len(value) > 8 else "SET"
        elif value is not None:
            results[var] = value
        else:
            results[var] = "NOT SET"
    return results


def main():
    print("=" * 60)
    print("  Paper 4: Environment Report")
    print("=" * 60)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": get_git_info(),
        "env_vars": check_env_vars(),
        "library_imports": check_imports(),
        "pip_freeze": get_pip_freeze(),
    }

    print(f"\nPython:   {sys.version.split()[0]}")
    print(f"OS:       {platform.system()} {platform.release()}")
    g = report["git"]
    print(f"Git:      {str(g['commit_hash'])[:8]}...")

    print("\n--- Environment Variables ---")
    for var, val in report["env_vars"].items():
        status = "OK" if val not in ("NOT SET",) else "MISSING"
        print(f"  {var}: {val} [{status}]")

    print("\n--- Library Imports ---")
    all_ok = True
    for lib, ver in report["library_imports"].items():
        if str(ver).startswith("MISSING"):
            print(f"  FAIL  {lib}: {ver}")
            all_ok = False
        else:
            print(f"  OK    {lib}: {ver}")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    report_path = logs_dir / "env_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    if all_ok:
        print("\nALL REQUIRED LIBRARIES AVAILABLE")
    else:
        print("\nWARNING: Some libraries missing. Run: pip install -r requirements.lock.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
