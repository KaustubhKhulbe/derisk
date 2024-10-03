import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from llama import Transformer, ModelArgs

import torch.profiler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time


MAX_SEQ_LEN = 2**4

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    from fairscale.nn.model_parallel import initialize_model_parallel
    initialize_model_parallel(4)

def cleanup():
    dist.destroy_process_group()

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
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

    torch.cuda.set_device(rank)

    transformer = Transformer(args).to(rank)
    # fsdp_transformer = FSDP(transformer)

    tensor = torch.rand(1, MAX_SEQ_LEN, device=rank).long()

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)

    prof.start()
    print("Hello World")
    prof.step()
    prof.stop()

    t = time.time()
    # fsdp_transformer(tensor, 0)
    # fsdp_transformer.forward(tensor, 0)


#    if rank == 0:
#        print(time.time() - t)
    cleanup()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
