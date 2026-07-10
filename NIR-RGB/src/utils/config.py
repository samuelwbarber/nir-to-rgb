from pathlib import Path
from types import SimpleNamespace
import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw), raw


def save_config(raw, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
