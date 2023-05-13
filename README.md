# Optimizating transformers
Simplified examples of common Transformer optimization techniques. These included
trade-offs from compute to memory (flash attention, checkpointing, caching), parallelization techniques for large language models (model, pipeline,
token, we can skip data here), stuff that generally improves training (QK normalization,
packing) and finally some common quantization approaches (int8, ...).

Where possible, I try to measure the increase in performance be it compute or memory. 
However, most examples are just that - examples to get a better understanding of 
how these techniques work. Parallelization techniques for example are written in MPI,
and it will not make sense to benchmark performance there.

| Technique | Implemented | Code | Unit Tests | Blog Post |
| :-------- | :---------: | :--- | :--------- | :-------- |
| Packing | ✔️ | [Packing](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/alibi.py) | [tests](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/alibi_test.py) |  |
| KV Cache | ✔️ | [KV Cache](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/kv_cache.py) | [tests](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/kv_cache_test.py) |  |
| Gradient Checkpointing | | | | |
| Flash Attention | | | | |
| Model Parallelization | | | | |
| Pipeline Parallelization | | | | |
| Token Parallelization | | | | |
| QK Normalization | ✔️ | [QK Normalization](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/qk_normalization.py) | [tests](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/qk_normalization_test.py) | |
| Int8 Quantization | | | | |
| ALiBi | ✔️ | [ALiBi](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/alibi.py) | | |
| RoPE | | | | |
