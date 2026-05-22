"""
Stage 13 Tier 2: PhoBERT title embeddings.

Encodes all item titles in dim_listing to 768-dim vectors using vinai/phobert-base.
Output: cache/item_title_emb.parquet (item_id + 768 float16 columns).

User text profile (built in 06_features.py): avg of pos-event item title embs per user.
"""
import sys, os, time, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{elapsed()} Device: {DEVICE}")

OUT_FILE = f"{CACHE_DIR}/item_title_emb.parquet"
if os.path.exists(OUT_FILE):
    print(f"{elapsed()} [SKIP] {OUT_FILE} already exists")
    raise SystemExit(0)

# ── Load item titles ──────────────────────────────────────────────────────────
print(f"{elapsed()} Loading dim_listing titles …")
import glob
dim_files = sorted(glob.glob(f"{DIM_DIR}/*.parquet"))
items_df = pd.concat([pd.read_parquet(f, columns=['item_id','category','title'])
                      for f in dim_files], ignore_index=True)
items_df = items_df.dropna(subset=['title']).drop_duplicates('item_id')
items_df = items_df[items_df['category'].isin([1010,1020,1030,1040,1050])]
items_df['title_length'] = items_df['title'].str.len().astype(np.int16)
print(f"{elapsed()} Items: {len(items_df):,}, title avg len: {items_df['title_length'].mean():.1f}")

# ── Load PhoBERT ──────────────────────────────────────────────────────────────
print(f"{elapsed()} Loading PhoBERT base …")
MODEL_NAME = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
HIDDEN_DIM = 768  # PhoBERT base hidden size
MAX_LEN = 96      # titles avg 55 chars, max 70 → 96 tokens safe

# ── Encode in batches ─────────────────────────────────────────────────────────
print(f"{elapsed()} Encoding {len(items_df):,} titles (batch_size=128, max_len={MAX_LEN}) …")
BATCH = 128
titles = items_df['title'].tolist()
item_ids = items_df['item_id'].tolist()
emb_buf = np.empty((len(titles), HIDDEN_DIM), dtype=np.float16)

with torch.no_grad():
    for start in range(0, len(titles), BATCH):
        end = min(start + BATCH, len(titles))
        batch_titles = titles[start:end]
        enc = tokenizer(batch_titles, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors='pt').to(DEVICE)
        outputs = model(**enc)
        # Use [CLS] pooling (first token last hidden state)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)
        emb_buf[start:end] = cls_emb
        if (start // BATCH) % 50 == 0:
            pct = 100 * end / len(titles)
            print(f"{elapsed()}   encoded {end:,}/{len(titles):,} ({pct:.1f}%)")
        del enc, outputs, cls_emb
        if start % (BATCH * 200) == 0:
            torch.cuda.empty_cache()

print(f"{elapsed()} Encoded all titles. Shape: {emb_buf.shape}")

# ── Save as parquet ───────────────────────────────────────────────────────────
emb_cols = [f"te_{i:03d}" for i in range(HIDDEN_DIM)]
emb_df = pd.DataFrame(emb_buf, columns=emb_cols)
emb_df['item_id']      = item_ids
emb_df['title_length'] = items_df['title_length'].values
emb_df = emb_df[['item_id','title_length'] + emb_cols]
emb_df.to_parquet(OUT_FILE, index=False)
print(f"{elapsed()} Saved {OUT_FILE}: {len(emb_df):,} items × {HIDDEN_DIM} dims  "
      f"({os.path.getsize(OUT_FILE)/1e6:.0f} MB)")
print(f"{elapsed()} DONE")
