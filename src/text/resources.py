from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd

DEFAULT_RUS_CEFR = Path("data/cefr/russian_cefr_sample.csv")


@lru_cache(maxsize=4)
def load_russian_cefr_mapping(path: str | Path = DEFAULT_RUS_CEFR) -> Mapping[str, str]:
    """Return a lemma → CEFR level mapping loaded from CSV."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Russian CEFR list not found at: {path}")
    csv = pd.read_csv(path)
    return {
        str(row.word).strip().lower(): str(row.level).strip()
        for _, row in csv.iterrows()
    }


__all__ = ["DEFAULT_RUS_CEFR", "load_russian_cefr_mapping"]
