# src/align/mutual_align.py
from typing import List, Set, Tuple
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel

# Use the awesome-align checkpoint (mBERT fine-tuned for alignment)
MODEL_NAME = "aneuraz/awesome-align-with-co"  # or "bert-base-multilingual-cased" as a fallback
_tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
_mdl = AutoModel.from_pretrained(MODEL_NAME)
_mdl.eval()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_mdl.to(_device)

def _tokenize_words(words: List[str]):
    enc = _tok(words, is_split_into_words=True, return_tensors="pt", return_attention_mask=True)
    word_ids = enc.word_ids(0)
    for k in enc:
        enc[k] = enc[k].to(_device)
    return enc, word_ids

def _layer_hs(enc, layer: int = 8):
    with torch.no_grad():
        out = _mdl(**enc, output_hidden_states=True)
    return out.hidden_states[layer].squeeze(0)  # [T,H]

def _pool_words(hs, word_ids: List[int]):
    buckets = defaultdict(list)
    for i, wid in enumerate(word_ids):
        if wid is None:  # specials
            continue
        buckets[wid].append(hs[i])
    keep = sorted(buckets.keys())
    reps = torch.stack([torch.stack(buckets[k]).mean(0) for k in keep])
    return reps, keep

def mutual_soft_align(kz_words: List[str], ru_words: List[str], layer:int=8, thresh:float=0.05) -> Set[Tuple[int,int]]:
    kz_enc, kz_wids = _tokenize_words(kz_words)
    ru_enc, ru_wids = _tokenize_words(ru_words)
    kz_hs = _layer_hs(kz_enc, layer)
    ru_hs = _layer_hs(ru_enc, layer)
    kz_rep, kz_keep = _pool_words(kz_hs, kz_wids)
    ru_rep, ru_keep = _pool_words(ru_hs, ru_wids)
    sim = kz_rep @ ru_rep.T
    p_rgk = torch.softmax(sim, -1)
    p_kgr = torch.softmax(sim, -2)
    links = set()
    for i in range(p_rgk.size(0)):
        for j in range(p_rgk.size(1)):
            if p_rgk[i, j] > thresh and p_kgr[i, j] > thresh:
                links.add((kz_keep[i], ru_keep[j]))
    return links
