from pathlib import Path
import yaml

_ROOT = Path(__file__).parent.parent

def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = _ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
