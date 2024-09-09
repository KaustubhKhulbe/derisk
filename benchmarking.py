import sys
from llama import Transformer, ModelArgs
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import torch
from torch.profiler import profile, record_function, ProfilerActivity

print("Starting!")

log_file = open("flex_attention.log", "w")
sys.stdout = log_file

torch.cuda.empty_cache()

dist.init_process_group(backend='nccl')

MAX_SEQ_LEN = 8192

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = ModelArgs(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128000, ffn_dim_multiplier=1.0, max_batch_size=1, max_seq_len=MAX_SEQ_LEN)
initialize_model_parallel(model_parallel_size_=1)  # Set the size of the tensor model parallel group

transformer = Transformer(args).to(device)

# Warm-up
tensor = torch.rand(1, MAX_SEQ_LEN, device="cuda").long()
for _ in range(5):
    transformer.forward(tensor, 0)

torch.cuda.synchronize()  # Ensure GPU operations are complete

# Profile
with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, with_flops=True) as prof:
    transformer.forward(tensor, 0)

torch.cuda.synchronize()  # Ensure GPU operations are complete

# Log profiler output
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

log_file.close()
dist.destroy_process_group()
