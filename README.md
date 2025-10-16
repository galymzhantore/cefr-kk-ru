# Kazakh → Russian CEFR Pipeline (Word & Text)

This repository provides a **clean, working prototype** that:
1. Translates Kazakh text → Russian.
2. Aligns Kazakh words/phrases to Russian words (phrase-aware).
3. Maps Russian words to **CEFR levels (A1–C2)** and assigns word-level difficulty to Kazakh words.
4. Aggregates word scores to estimate **text-level CEFR**.
5. (Optional) Trains a **Kazakh word-level CEFR classifier** from silver labels.

> Works out of the box with small sample data. Replace the sample Russian CEFR list with a larger one for real use.

## Quickstart

### 1) Conda (recommended)
```bash
conda env create -f environment.yml
conda activate kazakh_cefr_env
```

### 2) Or pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Download parallel corpora (KazParC) to `data/parallel`
```bash
python -m src.data.download_parallel
```

### 4) Build silver labels from sample parallel pairs
```bash
python -m src.pipeline.build_silver_labels
```

### 5) Run end-to-end demo (CLI)
```bash
python run_pipeline.py --text "Ол кітап оқып жатыр"
```

You should see:
- Russian translation
- Phrase alignments (e.g., "оқып жатыр" → "читает")
- CEFR levels per word/phrase
- Text-level CEFR estimate

### 6) Optional: Fine-tune awesome-align on Kazakh–Russian
Format a file `data/parallel/train.kazru` with lines: `kazakh ||| russian`, then:
```bash
bash scripts/train_align.sh
bash scripts/align_infer.sh  # produces word links
```

### 7) Optional: Train word-level classifier (Kazakh)
After building `data/labels/silver_word_labels.csv`:
```bash
python -m src.models.train_word
```

---

## Repo Layout
```
data/
  sample_parallel.csv
  cefr/russian_cefr_sample.csv
  parallel/              # downloaded corpora (KazParC etc.)
  labels/                # generated silver labels
models/
  word_cefr/             # fine-tuned classifier will be saved here
  text_cefr/
scripts/
  train_align.sh
  align_infer.sh
src/
  __init__.py
  align/
    __init__.py
    mutual_align.py
    merge_phrases.py
  data/
    __init__.py
    download_parallel.py
  pipeline/
    __init__.py
    build_silver_labels.py
  models/
    __init__.py
    train_word.py
    predict_word.py
  text/
    __init__.py
    predict_text.py
  translation/
    __init__.py
    translator.py
  utils.py
run_pipeline.py
tests/
  smoke_test.py
```

## Notes
- **Translation model:** `deepvk/kazRush-kk-ru` (HuggingFace).
- **Alignment (default in code):** mutual-soft alignment over mBERT embeddings (phrase merge).
  - You can switch to `awesome-align` via scripts for higher accuracy.
- **Russian CEFR list:** sample CSV provided. Replace with a larger curated list.
- **GPU:** If you have CUDA, edit `environment.yml` (remove `cpuonly`) and set a proper `cudatoolkit` version.

## License
For research purposes. Respect licenses of models and datasets you use.
