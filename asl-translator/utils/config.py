"""
config.py

Loads the project settings from config.yaml.

Why this file exists:
    Keeping settings in one YAML file (instead of scattered through the
    code) lets you change behaviour, such as the confidence threshold or
    the list of signs, without touching Python. Every script calls
    load_config() so they all agree on the same values.

Main functions:
    load_config()   -> reads config.yaml into a Python dictionary.
    resolve_path()  -> turns a path from the config into an absolute path.

Used by: app.py, training/train.py, training/evaluate.py,
training/prepare_islr.py and data_collection/collect_samples.py.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# The asl-translator/ folder. Relative paths in config.yaml are measured
# from here, so scripts work no matter which folder you run them from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Read the YAML settings file.

    Args:
        config_path: Optional path to a different YAML file. When omitted,
            the project's config.yaml is used.

    Returns:
        A nested dictionary, e.g. config["recognition"]["confidence_threshold"].

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    _check_sign_names(config, path)
    return config


def _check_sign_names(config: Dict[str, Any], path: Path) -> None:
    """
    Make sure every sign name in the vocabulary is text.

    Why: YAML turns unquoted words such as yes, no, on and off into
    true/false values, so "- yes" would silently become the boolean True
    and never match the "yes" label in the dataset.
    """
    signs = (config.get("vocabulary") or {}).get("signs") or []
    not_text = [sign for sign in signs if not isinstance(sign, str)]
    if not_text:
        raise ValueError(
            f"In {path}, vocabulary.signs contains {not_text}, which are not text. "
            'Put quotes around sign names such as "yes", "no", "on" or "off".'
        )


def resolve_path(path_from_config: Union[str, Path]) -> Path:
    """
    Convert a (possibly relative) path from config.yaml into an absolute path.

    Args:
        path_from_config: A path such as "data/samples".

    Returns:
        An absolute Path inside the project folder (absolute inputs are
        returned unchanged).
    """
    path = Path(path_from_config)
    return path if path.is_absolute() else PROJECT_ROOT / path
