from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import cefr_id_to_label

DEFAULT_MODEL_DIR = Path("models/transformer_word_cefr")
_CACHE: Dict[Tuple[str, str], tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]] = {}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _load_resources(
    model_dir: Path = DEFAULT_MODEL_DIR,
    *,
    device: str | torch.device | None = None,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Transformer CEFR model not found at '{path}'. "
            "Train it first via `python -m src.models.train_word_transformer`."
        )

    resolved_device = _resolve_device(device)
    cache_key = (str(path.resolve()), str(resolved_device))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(resolved_device)
    model.eval()
    _CACHE[cache_key] = (tokenizer, model, resolved_device)
    return tokenizer, model, resolved_device


def predict_transformer_word(
    word: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str | torch.device | None = None,
) -> str:
    tokenizer, model, resolved_device = _load_resources(model_dir, device=device)
    encoded = tokenizer(
        word.strip(),
        return_tensors="pt",
        truncation=True,
    ).to(resolved_device)
    with torch.no_grad():
        logits = model(**encoded).logits
        pred = int(torch.argmax(logits, dim=-1).item())
    return cefr_id_to_label(pred)
