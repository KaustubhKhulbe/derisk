import os
import time
import torch
from pytorch_memlab import profile
from llama import Transformer, ModelArgs

import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

dist.init_process_group(backend='nccl')

MAX_SEQ_LEN = 2 ** 4
rank = 0
device = torch.device(f"cuda:{rank}")

initialize_model_parallel(model_parallel_size_=4)

# Model arguments
args = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128000,
    ffn_dim_multiplier=1.0,
    max_batch_size=1,
    max_seq_len=MAX_SEQ_LEN
)

transformer = Transformer(args).to(device)
print("HERE")

@profile
def profile_run():
    start = time.time()
    for i in range(5):
        transformer.forward(tensor, 0)

    end = time.time()

# Warm-up
tensor = torch.rand(1, MAX_SEQ_LEN, device=device).long()
for _ in range(5):
    transformer.forward(tensor, 0)

torch.cuda.synchronize()

profile_run()

torch.cuda.synchronize()

