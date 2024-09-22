import os
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from pytorch_memlab import profile
from llama import Transformer, ModelArgs
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set the current process to use the correct GPU

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    MAX_SEQ_LEN = 2 ** 4
    device = torch.device(f"cuda:{rank}")

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

    # Initialize the transformer and wrap it with FSDP
    transformer = Transformer(args).to(device)
    transformer = FSDP(wrap(transformer), auto_wrap_policy=None)

    @profile
    def profile_run():
        print(f"[Rank {rank}] Running profile...")
        start = time.time()
        for i in range(5):
            transformer.forward(tensor, 0)

        end = time.time()
        print(f"[Rank {rank}] Average time: {(end - start) / 5}")

    # Warm-up
    tensor = torch.rand(1, MAX_SEQ_LEN, device=device).long()
    for _ in range(5):
        transformer.forward(tensor, 0)

    torch.cuda.synchronize()

    profile_run()

    torch.cuda.synchronize()

    cleanup()

if __name__ == "__main__":
    world_size = 4  # Number of GPUs to use
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

