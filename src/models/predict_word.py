import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import cefr_id_to_label

MODEL_DIR = "models/word_cefr"
_device = "cuda" if torch.cuda.is_available() else "cpu"

_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
_mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
_mdl.eval()

def predict_word(word: str) -> str:
    enc = _tok(word, return_tensors="pt", truncation=True).to(_device)
    with torch.no_grad():
        logits = _mdl(**enc).logits
        pred = int(torch.argmax(logits, -1).item())
    return cefr_id_to_label(pred)
