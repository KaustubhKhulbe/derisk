from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks, flex_attention, create_block_mask
import torch
from pytorch_memlab import profile
import time

flex_attention = torch.compile(flex_attention, dynamic=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1  # Batch size
HEADS = 8       # Number of attention heads
Q_LEN = 128     # Length of the query sequence
KV_LEN = 512    # Length of the key and value sequences
EMBED_DIM = 64  # Embedding dimension per head

def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= Q_LEN
    return causal_mask & window_mask

def sliding_window_causal2(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= Q_LEN // 16
    return causal_mask & window_mask

q = torch.randn(BATCH_SIZE, HEADS, Q_LEN, EMBED_DIM, dtype=torch.bfloat16).to(device)
k = torch.randn(BATCH_SIZE, HEADS, KV_LEN, EMBED_DIM, dtype=torch.bfloat16).to(device)
v = torch.randn(BATCH_SIZE, HEADS, KV_LEN, EMBED_DIM, dtype=torch.bfloat16).to(device)

block_mask = create_block_mask(sliding_window_causal2, 1, 1, Q_LEN, KV_LEN)

print("[PROFILE 1]")
for i in range(10): flex_attention(q, k, v, block_mask=block_mask)

@profile
def test():
    start = time.time()
    for i in range(5):
        output = flex_attention(q, k, v, block_mask=block_mask)
    end = time.time()

    print(f"AVG TIME: {(end - start) / 5}")

test()
