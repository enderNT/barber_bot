from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import BarbershopConfig


class BarbershopConfigLoader:
    def __init__(self, config_path: Path | str) -> None:
        self._config_path = Path(config_path)
        self._cached: BarbershopConfig | None = None

    def load(self) -> BarbershopConfig:
        if self._cached is None:
            with self._config_path.open("r", encoding="utf-8") as file:
                self._cached = BarbershopConfig.model_validate(json.load(file))
        return self._cached
