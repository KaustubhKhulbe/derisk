from llama import Transformer, ModelArgs
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import torch
from torch.profiler import profile, record_function, ProfilerActivity


torch.cuda.empty_cache()

dist.init_process_group(backend='nccl')

MAX_SEQ_LEN = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with profile(activities=[ProfilerActivity.CUDA],
    profile_memory=True, record_shapes=True) as prof:

    args = ModelArgs(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128000, ffn_dim_multiplier=1.0, max_batch_size=1, max_seq_len=MAX_SEQ_LEN)
    print("Starting!")
    initialize_model_parallel(model_parallel_size_=1)  # Set the size of the tensor model parallel group

    initial_memory = torch.cuda.memory_allocated()
    transformer = Transformer(args).to(device)
    final_memory = torch.cuda.memory_allocated()
    peak_memory_used = torch.cuda.max_memory_allocated()

    print(f"Memory allocated by function: {(final_memory - initial_memory) / (1024 ** 2):.2f} MB")
    print(f"Peak memory used by function: {peak_memory_used / (1024 ** 2):.2f} MB")

    torch.cuda.reset_peak_memory_stats()

    tensor = torch.rand(1, MAX_SEQ_LEN, device="cuda").long()

    torch.cuda.empty_cache()

    # Measure initial memory usage
    initial_memory = torch.cuda.memory_allocated()

    # Reset peak memory stats to start fresh measurement
    torch.cuda.reset_peak_memory_stats()

    transformer.forward(tensor, 0)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    # Measure memory usage after function call
    final_memory = torch.cuda.memory_allocated()
    peak_memory_used = torch.cuda.max_memory_allocated()

    # Print memory stats
    print(f"Memory allocated by function: {(final_memory - initial_memory) / (1024 ** 2):.2f} MB")
    print(f"Peak memory used by function: {peak_memory_used / (1024 ** 2):.2f} MB")

dist.destroy_process_group()
