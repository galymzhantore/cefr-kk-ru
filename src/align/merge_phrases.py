from typing import List, Tuple, Set
from collections import defaultdict

def merge_kz_to_single_ru(kz_words: List[str], ru_words: List[str], links:Set[tuple]):
    ru2kz = defaultdict(list)
    for i_kz, j_ru in links:
        ru2kz[j_ru].append(i_kz)
    merged = []
    for j_ru, kz_idxs in ru2kz.items():
        kz_sorted = sorted(kz_idxs)
        span = [kz_sorted[0]]
        for idx in kz_sorted[1:]:
            if idx == span[-1] + 1:
                span.append(idx)
            else:
                merged.append((tuple(span), j_ru))
                span = [idx]
        merged.append((tuple(span), j_ru))
    results = []
    for kz_span, j_ru in merged:
        kz_phrase = " ".join(kz_words[i] for i in kz_span)
        ru_word  = ru_words[j_ru]
        results.append((kz_phrase, ru_word, kz_span, j_ru))
    return results
