"""
config.py — Centralised configuration loader for HMM Bot.

All modules import `load_config()` from here.
No module should open settings.yaml directly.
"""

import os
import yaml

# Resolve the path relative to THIS file so it works from any cwd.
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")


def load_config(path: str = _SETTINGS_PATH) -> dict:
    """
    Load and return the full configuration dictionary from settings.yaml.

    Args:
        path: Absolute path to the YAML config file. Defaults to
              hmm_bot/config/settings.yaml.

    Returns:
        dict: Parsed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ── Convenience accessors ─────────────────────────────────────────────────────

def get_trading(config: dict) -> dict:
    return config["trading"]


def get_strategy(config: dict) -> dict:
    return config["strategy"]


def get_hmm(config: dict) -> dict:
    return config["hmm"]


def get_sessions(config: dict) -> dict:
    return config["sessions"]


def get_execution(config: dict) -> dict:
    return config["execution"]
