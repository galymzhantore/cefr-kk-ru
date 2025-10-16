import requests
import csv
import time

# Output file
OUTPUT = "openrussian_cefr_words.csv"

# Base API endpoint template
BASE_URL = "https://api.openrussian.org/api/wordlists/all?start={start}&level={level}&lang=en"

# Max start offsets for each CEFR level
LEVEL_LIMITS = {
    "A1": 50,
    "A2": 200,
    "B1": 450,
    "B2": 950,
    "C1": 1450,
    "C2": 1950,
}

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
}

def clean_word(word: str) -> str:
    """Remove apostrophes and lowercase the word."""
    return word.replace("'", "").strip().lower()

def fetch_words(level: str, start: int):
    """Fetch one page of words for a given CEFR level."""
    url = BASE_URL.format(start=start, level=level)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if not r.ok:
            print(f"❌ {level} start={start} failed ({r.status_code})")
            return []
        data = r.json()
        entries = data.get("result", {}).get("entries", [])
        words = []
        for entry in entries:
            word = entry.get("bare") or entry.get("accented")
            if word:
                words.append((clean_word(word), level))
        print(f"{level} start={start}: +{len(words)} words")
        return words
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return []

def main():
    all_words = []
    for level, max_start in LEVEL_LIMITS.items():
        print(f"\n=== Fetching level {level} ===")
        for start in range(0, max_start + 1, 50):
            all_words.extend(fetch_words(level, start))
            time.sleep(0.5)  # polite delay

    # Write to CSV
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["russian_word", "cefr_level"])
        writer.writerows(all_words)

    print(f"\n✅ Done! Saved {len(all_words)} words to {OUTPUT}")

if __name__ == "__main__":
    main()
