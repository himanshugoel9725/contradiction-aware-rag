"""
Environment setup — MUST be imported first by every script.

Sets environment variables that prevent accidental network calls to HuggingFace
model hub and datasets hub during experiment runs. Auto-sets on import so
`import src.paper4.utils.env_setup` is sufficient.

Why this exists:
    Without these env vars, the transformers library may silently download model
    weights during import, breaking reproducibility and wasting bandwidth.
    Setting TRANSFORMERS_OFFLINE=1 makes any such attempt raise an error immediately.

API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, NCBI_EMAIL) are loaded from a
.env file in the project root via python-dotenv. Copy .env.example to .env
and fill in your values.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Walk up from this file's location to find .env (handles any cwd)
    _env_file = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(dotenv_path=_env_file, override=False)
except ImportError:
    pass  # python-dotenv not installed yet; keys must be set manually


def setup_environment() -> None:
    """
    Set all required env vars for offline/reproducible operation.

    Sets:
        TRANSFORMERS_OFFLINE=1    — Prevents HuggingFace transformers from downloading
        HF_DATASETS_OFFLINE=1     — Prevents HuggingFace datasets from downloading
        TOKENIZERS_PARALLELISM=false — Suppresses tokenizer fork warnings
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def verify_offline() -> None:
    """
    Verify that all offline environment variables are correctly set.

    Raises:
        RuntimeError: If any required environment variable is missing or incorrect.
    """
    required = {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }

    missing = []
    for var, expected in required.items():
        actual = os.environ.get(var)
        if actual != expected:
            missing.append(f"  {var}: expected='{expected}', got='{actual}'")

    if missing:
        raise RuntimeError(
            "Environment verification failed:\n"
            + "\n".join(missing)
            + "\n\nFix: call setup_environment() first."
        )


# Auto-setup on import
setup_environment()
