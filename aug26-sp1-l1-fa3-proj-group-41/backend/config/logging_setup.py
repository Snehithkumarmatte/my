"""Logging setup. Everything goes to stdout - containers never write log files."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

_RULES_FILE = Path(__file__).resolve().parent / "rules.yaml"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def setup_logging() -> None:
    """Configure the root logger once. Level: LOG_LEVEL env, else rules.yaml, else INFO."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL")
    if not level_name:
        try:
            with _RULES_FILE.open(encoding="utf-8") as fh:
                level_name = ((yaml.safe_load(fh) or {}).get("logging") or {}).get("log_level")
        except (OSError, yaml.YAMLError):
            level_name = None
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # third-party libraries are very chatty at INFO
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


setup_loggin = setup_logging   # old (misspelt) name, kept so nothing breaks
