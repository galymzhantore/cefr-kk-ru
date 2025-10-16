from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

from src.align.merge_phrases import merge_kz_to_single_ru
from src.align.mutual_align import EmbeddingAligner, get_default_aligner
from src.translation.translator import Translator, get_translator
from src.text.resources import DEFAULT_RUS_CEFR, load_russian_cefr_mapping
CEFR_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_WEIGHTS = {level: idx for idx, level in enumerate(CEFR_ORDER)}


def _tokenize_words(text: str) -> Sequence[str]:
    return tuple(w for w in text.strip().split() if w)


def predict_text_cefr(
    kaz_text: str,
    rus_cefr_path: str | Path = DEFAULT_RUS_CEFR,
    translator: Translator | None = None,
    aligner: EmbeddingAligner | None = None,
):
    translator = translator or get_translator()
    aligner = aligner or get_default_aligner()

    translation = translator.translate(kaz_text)
    kz_words = _tokenize_words(kaz_text)
    ru_words = _tokenize_words(translation)

    links = aligner.align(kz_words, ru_words, layer=8, thresh=0.05)
    phrases = merge_kz_to_single_ru(kz_words, ru_words, links)

    russian_levels = load_russian_cefr_mapping(rus_cefr_path)
    counts = Counter()
    for _, ru_word, _, _ in phrases:
        level = russian_levels.get(ru_word.lower(), "Unknown")
        counts[level] += 1

    known_total = sum(counts[level] for level in CEFR_ORDER)
    if known_total:
        distribution = {level: counts[level] / known_total for level in CEFR_ORDER}
        avg = sum(CEFR_WEIGHTS[level] * distribution[level] for level in CEFR_ORDER)
        avg_level = CEFR_ORDER[round(avg)]
    else:
        distribution = {level: 0.0 for level in CEFR_ORDER}
        avg_level = "Unknown"

    return {
        "translation": translation,
        "distribution": distribution,
        "avg_level": avg_level,
        "phrases": phrases,
    }
