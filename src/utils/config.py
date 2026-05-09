"""Configuration helpers with a YAML fallback parser.

The project intentionally avoids requiring PyYAML in stage one so that
inference can run in the existing DL2 environment without new installs.
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file.

    PyYAML is used when available. If it is not installed, a small parser
    handles the limited YAML subset used by the project's config files.
    """

    path = Path(config_path)
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        return data
    except ModuleNotFoundError:
        return _parse_simple_yaml(text, path)


def _parse_simple_yaml(text: str, path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML syntax at {path}:{lineno}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if not value:
            child: Dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        current[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]

    if value.startswith("[") and value.endswith("]"):
        parsed = literal_eval(value)
        return parsed

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value
