import argparse
from src.text.predict_text import predict_text_cefr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Kazakh text")
    args = ap.parse_args()
    out = predict_text_cefr(args.text)
    print("Translation:", out["translation"])
    print("Text CEFR:", out["avg_level"])
    print("Distribution:", out["distribution"])
    print("Phrase alignments (KZ → RU):")
    for kz_phrase, ru_word, _, _ in out["phrases"]:
        print(f'  "{kz_phrase}" -> "{ru_word}"')

if __name__ == "__main__":
    main()
