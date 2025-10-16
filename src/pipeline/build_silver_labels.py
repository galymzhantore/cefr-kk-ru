from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import inspect

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _compat_getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)

    inspect.getargspec = _compat_getargspec  # type: ignore[attr-defined]

import pandas as pd

try:
    import pymorphy3 as pymorphy
except ImportError:  # pragma: no cover
    import pymorphy2 as pymorphy

from src.align.mutual_align import EmbeddingAligner, get_default_aligner
from src.data.repositories import DEFAULT_RUS_CEFR, RussianCefrRepository
from src.domain.services import AlignmentService

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
    repository = RussianCefrRepository(rus_cefr)
    mapping = repository.mapping
    aligner = aligner or get_default_aligner()
    alignment_service = AlignmentService(
        aligner, default_layer=layer, default_threshold=thresh
    )

    rows: list[dict[str, str]] = []
    skipped_sequences = 0
    for sample in df.itertuples(index=False):
        kz = str(getattr(sample, "kaz")).strip()
        ru = str(getattr(sample, "rus")).strip()
        kz_words = tuple(part for part in kz.split() if part)
        ru_words = tuple(part for part in ru.split() if part)
        phrases = alignment_service.align_phrases(kz_words, ru_words)
        if not phrases:
            skipped_sequences += 1
            continue
        for phrase in phrases:
            lemma = morph.parse(phrase.russian_token.lower())[0].normal_form
            level = mapping.get(
                lemma, mapping.get(phrase.russian_token.lower(), "Unknown")
            )
            rows.append(
                {
                    "kaz_item": phrase.kazakh_phrase,
                    "rus_item": phrase.russian_token,
                    "cefr": level,
                    "kaz_sent": kz,
                    "rus_sent": ru,
                }
            )

    output_df = pd.DataFrame(rows, columns=["kaz_item", "rus_item", "cefr", "kaz_sent", "rus_sent"])
    output_df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"Saved: {out_csv} rows={len(rows)} skipped_sentences={skipped_sequences}")
    return out_csv


if __name__ == "__main__":
    main()
