from datasets import load_dataset
import os, csv

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def save_kz_ru(split="train", out_dir="data/parallel", out_name="kazparc_kz_ru.csv"):
    ensure_dir(out_dir)
    ds = load_dataset("issai/kazparc", split=split)
    path = os.path.join(out_dir, out_name)
    with open(path, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kaz","rus"])
        for ex in ds:
            kk = ex.get("kk"); ru = ex.get("ru")
            if kk and ru:
                kk = kk.replace("\n"," ").strip()
                ru = ru.replace("\n"," ").strip()
                w.writerow([kk, ru])
    print("Saved:", path, "rows:", sum(1 for _ in open(path, encoding="utf-8"))-1)

if __name__ == "__main__":
    save_kz_ru()
