from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

try:
    import pymorphy3 as pymorphy
except ImportError:  # pragma: no cover
    import pymorphy2 as pymorphy

from src.align.mutual_align import EmbeddingAligner, get_default_aligner
from src.align.merge_phrases import merge_kz_to_single_ru
from src.text.resources import DEFAULT_RUS_CEFR, load_russian_cefr_mapping

PARALLEL_CSV = Path("data/parallel/kazparc_kz_ru.csv")  # or your own pairs
RUS_CEFR = DEFAULT_RUS_CEFR
OUT_CSV = Path("data/labels/silver_word_labels.csv")

morph = pymorphy.MorphAnalyzer()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(
    parallel_csv: str | Path = PARALLEL_CSV,
    rus_cefr: str | Path = RUS_CEFR,
    out_csv: str | Path = OUT_CSV,
    aligner: EmbeddingAligner | None = None,
    layer: int = 8,
    thresh: float = 0.05,
) -> Path:
    parallel_csv = Path(parallel_csv)
    rus_cefr = Path(rus_cefr)
    out_csv = Path(out_csv)

    ensure_dir(out_csv.parent)

    df = pd.read_csv(parallel_csv)
    russian_levels = load_russian_cefr_mapping(rus_cefr)
    aligner = aligner or get_default_aligner()

    rows: list[list[str]] = []
    for sample in df.itertuples(index=False):
        kz = str(getattr(sample, "kaz")).strip()
        ru = str(getattr(sample, "rus")).strip()
        kz_words = tuple(kz.split())
        ru_words = tuple(ru.split())
        links = aligner.align(kz_words, ru_words, layer=layer, thresh=thresh)
        phrases = merge_kz_to_single_ru(kz_words, ru_words, links)
        for kz_phrase, ru_word, _, _ in phrases:
            lemma = morph.parse(ru_word.lower())[0].normal_form
            level = russian_levels.get(lemma, russian_levels.get(ru_word.lower(), "Unknown"))
            rows.append([kz_phrase, ru_word, level, kz, ru])

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kaz_item", "rus_item", "cefr", "kaz_sent", "rus_sent"])
        writer.writerows(rows)

    print(f"Saved: {out_csv} rows={len(rows)}")
    return out_csv


if __name__ == "__main__":
    main()
