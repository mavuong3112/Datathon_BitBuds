"""
Step 4: SASRec — Sequential Self-Attention Recommendation (PyTorch GPU)
80 epochs, memory-efficient chunked inference to avoid OOM with 691K items.
Outputs:
  cache/sasrec_candidates.parquet
  cache/sasrec_model.pt
"""
import sys, os, time, pickle, gc
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import *

t0 = time.time()
def elapsed(): return f"[{time.time()-t0:.0f}s]"
def mem_mb():
    import psutil; return psutil.Process().memory_info().rss / 1e6

OUT_PATH   = f"{CACHE_DIR}/sasrec_candidates.parquet"
MODEL_PATH = f"{CACHE_DIR}/sasrec_model.pt"

if os.path.exists(OUT_PATH):
    try:
        _df = pd.read_parquet(OUT_PATH)
        if len(_df) > 0:
            print(f"{elapsed()} [SKIP] sasrec_candidates.parquet exists ({len(_df):,} rows)")
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{elapsed()} Device: {DEVICE}")
if DEVICE == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"{elapsed()} GPU: {props.name}, VRAM: {props.total_memory/1e9:.1f}GB")

# ── Hyperparams ───────────────────────────────────────────────────────────────
MAX_SEQ_LEN  = 50
N_LAYERS     = 2
N_HEADS      = 4
HIDDEN_DIM   = 128
DROPOUT      = 0.2
TRAIN_BATCH  = 2048      # reduced to save VRAM during training
INFER_BATCH  = 512       # users per inference batch
ITEM_CHUNK   = 80_000    # items per scoring chunk (avoid full n_items×INFER_BATCH matrix)
EPOCHS       = 80
LR           = 1e-3

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"{elapsed()} Loading data …  [RAM:{mem_mb():.0f}MB]")
# Only load needed columns to save RAM
seq  = pd.read_parquet(f"{CACHE_DIR}/user_item_seq.parquet",
                       columns=['user_id', 'item_id', 'event_ts'])
test = pd.read_parquet(TEST_FILE)
with open(f"{CACHE_DIR}/mappings.pkl", 'rb') as f:
    maps = pickle.load(f)
user2idx  = maps['user2idx']
item2idx  = maps['item2idx']
idx2item  = maps['idx2item']
del maps
test_users    = set(test['user_id'].tolist())
del test
test_in_train = [u for u in test_users if u in user2idx]
n_items = len(item2idx) + 1   # +1 for pad token 0
print(f"{elapsed()} n_items={n_items:,}  warm_test={len(test_in_train):,}  [RAM:{mem_mb():.0f}MB]")

# ── Build per-user item sequences ─────────────────────────────────────────────
print(f"{elapsed()} Building sequences …")
# Use merge instead of .map() — avoids ArrowDtype object-array allocation failure
mapping_df = pd.DataFrame({'item_id': list(item2idx.keys()),
                            'item_idx': list(item2idx.values())})
seq = seq.merge(mapping_df, on='item_id', how='inner')
del mapping_df; gc.collect()
seq['item_idx'] = seq['item_idx'].astype(np.int32) + 1   # 0=pad

seq = seq.sort_values(['user_id', 'event_ts'])
user_seqs = seq.groupby('user_id', sort=False)['item_idx'].apply(list).to_dict()
del seq; gc.collect()   # free 1.3 GB — no longer needed after dict is built
print(f"{elapsed()} Sequences built: {len(user_seqs):,} users  [RAM:{mem_mb():.0f}MB]")

def pad_seq(s, maxlen):
    s = s[-maxlen:]
    return [0] * (maxlen - len(s)) + s

# ── Dataset ───────────────────────────────────────────────────────────────────
class SASRecDataset(Dataset):
    def __init__(self, user_seqs, maxlen):
        self.data   = [(u, s) for u, s in user_seqs.items() if len(s) >= 2]
        self.maxlen = maxlen

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        uid, items = self.data[idx]
        inp = pad_seq(items[:-1], self.maxlen)
        tgt = pad_seq(items[1:],  self.maxlen)
        neg = np.random.randint(1, n_items, size=self.maxlen).tolist()
        return (torch.LongTensor(inp),
                torch.LongTensor(tgt),
                torch.LongTensor(neg))

# ── SASRec Model ──────────────────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, n_items, hidden, n_layers, n_heads, maxlen, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, hidden, padding_idx=0)
        self.pos_emb  = nn.Embedding(maxlen + 1, hidden)
        self.emb_drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                              enable_nested_tensor=False)
        self.ln      = nn.LayerNorm(hidden)
        self.maxlen  = maxlen

    def encode(self, x):
        B, L = x.shape
        pos  = torch.arange(1, L + 1, device=x.device).unsqueeze(0).expand(B, -1)
        # Zero out positions of padding tokens
        pos  = pos * (x != 0).long()
        h    = self.item_emb(x) + self.pos_emb(pos)
        h    = self.emb_drop(h)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        h    = self.encoder(h, mask=mask)
        return self.ln(h)   # (B, L, hidden)

    def forward(self, x, pos_items, neg_items):
        h      = self.encode(x)          # (B, L, hidden)
        pos_e  = self.item_emb(pos_items)  # (B, L, hidden)
        neg_e  = self.item_emb(neg_items)
        pos_sc = (h * pos_e).sum(-1)     # (B, L)
        neg_sc = (h * neg_e).sum(-1)
        return pos_sc, neg_sc

    def get_last_hidden(self, x):
        h = self.encode(x)               # (B, L, hidden)
        # Find last non-padding position
        lengths = (x != 0).sum(dim=1) - 1   # (B,) — index of last real token
        lengths = lengths.clamp(min=0)
        idx     = lengths.unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(2))
        return h.gather(1, idx).squeeze(1)  # (B, hidden)

# ── Train ─────────────────────────────────────────────────────────────────────
train_users = {u: s for u, s in user_seqs.items() if len(s) >= 2}
dataset = SASRecDataset(train_users, MAX_SEQ_LEN)
# pin_memory=False: safer when system RAM is fragmented / under pressure
loader  = DataLoader(dataset, batch_size=TRAIN_BATCH, shuffle=True,
                     num_workers=0, pin_memory=False)

model = SASRec(n_items, HIDDEN_DIM, N_LAYERS, N_HEADS, MAX_SEQ_LEN, DROPOUT).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
bce   = nn.BCEWithLogitsLoss(reduction='none')

if os.path.exists(MODEL_PATH):
    print(f"{elapsed()} [SKIP training] Loading saved model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print(f"{elapsed()} Training SASRec ({EPOCHS} epochs, {len(dataset):,} users, batch={TRAIN_BATCH}) …")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss, n_b = 0.0, 0
        for inp, tgt, neg in loader:
            inp, tgt, neg = inp.to(DEVICE), tgt.to(DEVICE), neg.to(DEVICE)
            pos_sc, neg_sc = model(inp, tgt, neg)
            mask  = (tgt != 0).float()
            loss  = (bce(pos_sc, torch.ones_like(pos_sc)) +
                     bce(neg_sc, torch.zeros_like(neg_sc))) * mask
            loss  = loss.sum() / mask.sum().clamp(min=1)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += loss.item(); n_b += 1

        sched.step()
        if epoch % 10 == 0 or epoch == EPOCHS:
            lr_now = sched.get_last_lr()[0]
            print(f"{elapsed()} Epoch {epoch:3d}/{EPOCHS}  loss={ep_loss/n_b:.4f}  lr={lr_now:.2e}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"{elapsed()} Model saved  [RAM:{mem_mb():.0f}MB]")

# ── Inference: chunked item scoring to avoid OOM ─────────────────────────────
print(f"{elapsed()} Generating SASRec candidates (chunked, {ITEM_CHUNK:,} items/chunk) …")
model.eval()

# Pre-compute ALL item embeddings once (n_items × hidden: 691K×128×4 = ~354MB — fine)
with torch.no_grad():
    all_item_ids  = torch.arange(1, n_items, device=DEVICE)      # skip pad
    all_item_emb  = model.item_emb(all_item_ids)                  # (n_items-1, H)

rows_sr = []

with torch.no_grad():
    for i in range(0, len(test_in_train), INFER_BATCH):
        batch_users = test_in_train[i:i + INFER_BATCH]
        seqs = [pad_seq(user_seqs.get(u, [1]), MAX_SEQ_LEN) for u in batch_users]
        inp  = torch.LongTensor(seqs).to(DEVICE)        # (B, L)
        h    = model.get_last_hidden(inp)                # (B, H)

        # Score in item chunks to keep memory < INFER_BATCH × ITEM_CHUNK × 4 bytes
        # = 512 × 80000 × 4 = 164 MB — safe
        n_real_items = all_item_emb.size(0)
        best_scores = torch.full((len(batch_users), N_SASREC), float('-inf'), device=DEVICE)
        best_ids    = torch.zeros((len(batch_users), N_SASREC), dtype=torch.long, device=DEVICE)

        for c_start in range(0, n_real_items, ITEM_CHUNK):
            c_end    = min(c_start + ITEM_CHUNK, n_real_items)
            chunk_e  = all_item_emb[c_start:c_end]          # (chunk, H)
            chunk_sc = h @ chunk_e.T                          # (B, chunk)
            # Merge with running top-K
            full_sc  = torch.cat([best_scores, chunk_sc], dim=1)    # (B, N+chunk)
            full_id  = torch.cat([best_ids,
                                  torch.arange(c_start + 1, c_end + 1,   # +1 for pad shift
                                               device=DEVICE).unsqueeze(0).expand(len(batch_users), -1)
                                 ], dim=1)
            topk     = torch.topk(full_sc, N_SASREC, dim=1)
            best_scores = topk.values
            best_ids    = full_id.gather(1, topk.indices)

        # One bulk GPU→CPU transfer instead of per-element .item() syncs
        best_ids_np    = best_ids.cpu().numpy()     # (B, N_SASREC)
        best_scores_np = best_scores.cpu().numpy()  # (B, N_SASREC)
        for j, uid in enumerate(batch_users):
            for rank in range(N_SASREC):
                iid = int(best_ids_np[j, rank])
                sc  = float(best_scores_np[j, rank])
                if 1 <= iid < n_items and iid in idx2item:
                    rows_sr.append({'user_id': uid, 'item_id': idx2item[iid],
                                    'sasrec_score': sc, 'sasrec_rank': rank + 1})

        if (i // INFER_BATCH) % 20 == 0:
            print(f"{elapsed()} SASRec inference {i//INFER_BATCH+1}/{(len(test_in_train)-1)//INFER_BATCH+1}")

df_sr = pd.DataFrame(rows_sr)
print(f"{elapsed()} SASRec candidates: {len(df_sr):,} rows")
df_sr.to_parquet(f"{CACHE_DIR}/sasrec_candidates.parquet", index=False)
print(f"{elapsed()} DONE")
