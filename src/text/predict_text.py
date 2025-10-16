from collections import Counter
from src.translation.translator import Translator
from src.align.mutual_align import mutual_soft_align
from src.align.merge_phrases import merge_kz_to_single_ru
import pandas as pd

RUS_CEFR = "data/cefr/russian_cefr.csv"

def predict_text_cefr(kaz_text: str):
    tr = Translator()
    ru = tr.translate(kaz_text)
    kz_words = kaz_text.strip().split()
    ru_words = ru.strip().split()
    links = mutual_soft_align(kz_words, ru_words, layer=8, thresh=0.05)
    phrases = merge_kz_to_single_ru(kz_words, ru_words, links)

    rc = pd.read_csv(RUS_CEFR)
    r2c = {r.word.strip().lower(): r.level for _, r in rc.iterrows()}

    from collections import Counter
    counts = Counter()
    for kz_phrase, ru_word, _, _ in phrases:
        lvl = r2c.get(ru_word.lower(), "Unknown")
        counts[lvl] += 1

    order = ["A1","A2","B1","B2","C1","C2"]
    total = sum(counts.values()) or 1
    dist = {k: counts.get(k,0)/total for k in order}
    weights = {k:i for i,k in enumerate(order)}
    avg = sum(weights[k]*v for k,v in dist.items())
    avg_level = order[round(avg)]
    return {"translation": ru, "distribution": dist, "avg_level": avg_level, "phrases": phrases}
