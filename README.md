# Regular Attention Derisking

### `benchmarking.py`
1. define maximum sequence length
2. initialize new transformer model with said seq len
3. create random tensor with said seq len
4. run tensor through forward pass in transformer
5. output ram usage

### Modifications
##### `llama/model.py`
- Under `Feedforward` init function, change `hidden_dim` to `hidden_dim = 14336`, the official Llama 8B dimensions
- Create `flex_attention` based function for causal and sliding window masks
- Apply the masks in `forward` in the `Attention` class

### Issues
- RAM Usage does not appear to change when sparsity level (Sliding Window Dimensions) change
