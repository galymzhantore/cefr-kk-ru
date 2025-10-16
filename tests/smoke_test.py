from src.translation.translator import Translator
from src.align.mutual_align import mutual_soft_align
from src.align.merge_phrases import merge_kz_to_single_ru

def test_smoke():
    kz = "Ол кітап оқып жатыр"
    ru = "Он читает книгу"
    tr = Translator()
    out = tr.translate(kz)
    assert isinstance(out, str) and len(out) > 0

    kz_words = kz.split(); ru_words = ru.split()
    links = mutual_soft_align(kz_words, ru_words, layer=8, thresh=0.05)
    phrases = merge_kz_to_single_ru(kz_words, ru_words, links)
    assert isinstance(phrases, list)
    print("OK")
