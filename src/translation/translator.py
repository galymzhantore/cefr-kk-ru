import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TranslationPipeline

MODEL_NAME = "issai/tilmash"

class Translator:
    def __init__(self, device: str | None = None):
        # device index for pipeline: 0..N for CUDA, -1 for CPU
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # NLLB/Tilmash supports these codes per model card:
        # kaz_Cyrl (Kazakh), rus_Cyrl (Russian)
        self.pipe = TranslationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang="kaz_Cyrl",
            tgt_lang="rus_Cyrl",
            max_length=1000,
            device=self.device,
        )

    def translate(self, text: str) -> str:
        return self.pipe(text)[0]["translation_text"]
