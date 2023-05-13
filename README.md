# Optimizating transformers
Simplified examples of common Transformer optimization techniques. These included
trade-offs from compute to memory (flash attention, checkpointing, caching), parallelization techniques for large language models (model, pipeline,
token, we can skip data here), stuff that generally improves training (QK normalization,
packing) and finally some common quantization approaches (int8, ...).

Where possible, I try to measure the increase in performance be it compute or memory. 
However, most examples are just that - examples to get a better understanding of 
how these techniques work. Parallelization techniques for example are written in MPI,
and it will not make sense to benchmark performance there.

- [X] [Packing](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/alibi.py) | [tests](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/alibi_test.py)
- [X] [KV-Cache](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/kv_cache.py) | [tests](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/kv_cache_test.py)
- [ ] Gradient Checkpointing / Rematerialization / activation recomputation
- [ ] Flash Attention
- [ ] Model Parallization
- [ ] Pipeline Parallelization
- [ ] Token parallelization
- [X] [QK Normalization](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/qk_normalization.py) | [tests](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/qk_normalization_test.py)
- [ ] Int8 Quantization
- [X] [ALiBi](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/alibi.py) | [tests](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/alibi_test.py)
- [ ] RoPE