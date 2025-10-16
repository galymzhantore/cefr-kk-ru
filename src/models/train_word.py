import pandas as pd, numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from src.utils import set_seed, cefr_label_to_id

CSV_IN = "data/labels/silver_word_labels.csv"
MODEL_NAME = "kz-transformers/kaz-roberta-conversational"
OUT_DIR = "models/word_cefr"
NUM_LABELS = 6

def load_data(csv_path=CSV_IN):
    df = pd.read_csv(csv_path)
    df = df[df["cefr"].isin(["A1","A2","B1","B2","C1","C2"])]
    if df.empty:
        raise RuntimeError("No labeled rows found. Expand russian_cefr_sample.csv first.")
    df["label"] = df["cefr"].apply(cefr_label_to_id)
    df["text"] = df["kaz_item"].astype(str)
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    return DatasetDict({
        "train": Dataset.from_pandas(tr.reset_index(drop=True)),
        "test": Dataset.from_pandas(te.reset_index(drop=True))
    })

def main():
    set_seed(42)
    ds = load_data()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tok_fun(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=64)
    ds = ds.map(tok_fun, batched=True)
    ds = ds.rename_column("label","labels")
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    args = TrainingArguments(
        output_dir=OUT_DIR, per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=3, evaluation_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", logging_steps=20
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        from sklearn.metrics import f1_score
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"], tokenizer=tok, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(OUT_DIR)
    print("Saved to", OUT_DIR)

if __name__ == "__main__":
    main()
