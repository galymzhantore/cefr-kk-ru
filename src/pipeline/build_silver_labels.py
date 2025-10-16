# top of file
import csv, os, pandas as pd
# at the very top
try:
    import pymorphy3 as pymorphy
except ImportError:
    import pymorphy2 as pymorphy
morph = pymorphy.MorphAnalyzer()

from src.align.mutual_align import mutual_soft_align
from src.align.merge_phrases import merge_kz_to_single_ru

PARALLEL_CSV = "data/parallel/kazparc_kz_ru.csv"  # or your own pairs
RUS_CEFR = "data/cefr/russian_cefr_full.csv"
OUT_CSV = "data/labels/silver_word_labels.csv"
morph = pymorphy.MorphAnalyzer()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main(parallel_csv=PARALLEL_CSV, rus_cefr=RUS_CEFR, out_csv=OUT_CSV):
    ensure_dir(os.path.dirname(out_csv))
    df = pd.read_csv(parallel_csv)
    rc = pd.read_csv(rus_cefr)
    # normalize to lemmas
    r2c = {str(r.word).strip().lower(): str(r.level).strip() for _, r in rc.iterrows()}

    rows = []
    for _, row in df.iterrows():
        kz = str(row["kaz"]).strip()
        ru = str(row["rus"]).strip()
        kz_words = kz.split()
        ru_words = ru.split()
        links = mutual_soft_align(kz_words, ru_words, layer=8, thresh=0.05)
        phrases = merge_kz_to_single_ru(kz_words, ru_words, links)
        for kz_phrase, ru_word, _, _ in phrases:
            lemma = morph.parse(ru_word.lower())[0].normal_form
            level = r2c.get(lemma, r2c.get(ru_word.lower(), "Unknown"))
            rows.append([kz_phrase, ru_word, level, kz, ru])

    with open(out_csv, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kaz_item","rus_item","cefr","kaz_sent","rus_sent"])
        w.writerows(rows)
    print(f"Saved: {out_csv}  rows: {len(rows)}")
