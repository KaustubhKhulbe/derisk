import os
import click
from math import ceil

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ring_attention_pytorch.ring_attention import RingTransformer
from ring_attention_pytorch.distributed import all_gather_variable_dim

from pytorch_memlab import profile
import torch.profiler
from torch.profiler import ProfilerActivity

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("trace_" + str(p.step_num) + ".json")

prof = torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                                on_trace_ready=trace_handler)


            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/ring'),
            # record_shapes=True,
            # with_stack=True

def setup(
    rank,
    world_size,
    use_cuda
):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    backend = "nccl"
    dist.init_process_group(backend, rank = rank, world_size = world_size)

    
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def start(
    rank,
    world_size, batch_size, num_sharded_batches, batch_size_var_len, use_cuda, causal,
    striped_ring_attn, num_buckets, seq_len, model_dim, heads, num_grouped_query_heads,
    dim_head, num_tokens, depth, ff_mult, ring_seq_size, bucket_size, rotary_embed_theta,
    force_regular_attn
):

    torch.cuda.synchronize()
    prof.start()
    setup(rank, world_size, use_cuda)

    ring_seq_size = ceil(seq_len / world_size) * num_sharded_batches
    bucket_size = ring_seq_size // num_buckets

    print("Building...")
    ring_attention = RingTransformer(
        dim = model_dim,
        depth = depth,
        causal = causal,
        dim_head = dim_head,
        heads = heads,
        num_grouped_query_heads = num_grouped_query_heads,
        ring_attn = True,
        striped_ring_attn = False,
        ring_seq_size = ring_seq_size,
        bucket_size = bucket_size,
        use_cuda_kernel = use_cuda,
        auto_shard_seq = True,
        num_tokens=num_tokens,
        ff_mult = 3,
    )

    seq = torch.rand(1, seq_len, device=rank).long()

    seq = seq.cuda(rank)
    ring_attention.cuda(rank)

    ring_input = seq.clone()

    ddp_ring_attention = DDP(ring_attention, device_ids=[rank])

    print("Running Attention Mechanism")

    prof.step()
    ddp_ring_attention.forward(ring_input)

    prof.step()
    ddp_ring_attention.forward(ring_input)
    
    if rank == 0:

        ring_attention = ring_attention.cpu()
        print('✅ Ran Ring Attention')

    torch.cuda.synchronize()
    prof.stop()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    cleanup()

@click.command()
@click.option('--world-size', default=4, help='number of machines / processes')
@click.option('--batch-size', default=1, help='test batch size (matches max_batch_size from ModelArgs)')
@click.option('--num-sharded-batches', default=1, help='number of sharded batches')
@click.option('--batch-size-var-len', is_flag=False, help='test variable lengthed batch sizes')
@click.option('--use-cuda', is_flag=True, help='whether to test with CUDA and NCCL')
@click.option('--causal', is_flag=True, help='test autoregressive')
@click.option('--striped-ring-attn', is_flag=False, help='test striped ring attention from MIT follow up paper')
@click.option('--num-buckets', default=2, help='number of buckets per machine (each sharded sequence is further windowed for flash attention to achieve even greater context lengths)')
@click.option('--seq-len', default=1024, help='sequence length to test (matches max_seq_len from ModelArgs)')
@click.option('--model-dim', default=4096, help='model dimensions (matches dim from ModelArgs)')
@click.option('--heads', default=32, help='number of query attention heads (matches n_heads from ModelArgs)')
@click.option('--num-grouped-query-heads', default=4, help='number of query attention head groups (matches num_grouped_query_heads)')
@click.option('--dim-head', default=64, help='model dimensions for testing (matches dim_head)')
@click.option('--num-tokens', default=128000, help='number of tokens (matches vocab_size from ModelArgs)')
@click.option('--depth', default=32, help='depth of the model (number of layers)')
@click.option('--ff-mult', default=3.5, help='feedforward multiplier (matches ff_mult from ModelArgs)')
@click.option('--ring-seq-size', default=512, help='ring sequence size (matches ring_seq_size from ModelArgs)')
@click.option('--bucket-size', default=512, help='bucket size for attention (matches bucket_size from ModelArgs)')
@click.option('--rotary-embed-theta', default=10000, help='rotary embedding theta (matches rotary_embed_theta from ModelArgs)')
@click.option('--force-regular-attn', is_flag=False, help='force regular attention instead of any specialized attention')

def test(world_size, batch_size, num_sharded_batches, batch_size_var_len, use_cuda, causal,
         striped_ring_attn, num_buckets, seq_len, model_dim, heads, num_grouped_query_heads,
         dim_head, num_tokens, depth, ff_mult, ring_seq_size, bucket_size, rotary_embed_theta,
         force_regular_attn):
    assert not use_cuda or world_size <= torch.cuda.device_count(), f'world size {world_size} must be less than the number of cuda devices {torch.cuda.device_count()}'

    mp.spawn(
        start,
        args = (
            world_size, batch_size, num_sharded_batches, batch_size_var_len, use_cuda, causal,
         striped_ring_attn, num_buckets, seq_len, model_dim, heads, num_grouped_query_heads,
         dim_head, num_tokens, depth, ff_mult, ring_seq_size, bucket_size, rotary_embed_theta,
         force_regular_attn
        ),
        nprocs = world_size,
        join = True
    )

if __name__ == '__main__':
    torch.cuda.synchronize()
    test()
    torch.cuda.synchronize()