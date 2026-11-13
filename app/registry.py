from pathlib import Path
from typing import Protocol

import yaml

from app.domain import ModelSpec


class ModelRegistry(Protocol):
    def list_models(self) -> list[ModelSpec]: ...


class YamlModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def list_models(self) -> list[ModelSpec]:
        if not self.path.exists():
            return []
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return [ModelSpec.model_validate(item) for item in raw.get("models", [])]
