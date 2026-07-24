# KV Cache Calculator

一个面向 LLM Inference / AI Infra 学习者和部署前 sizing 的静态网页工具。当前支持两个模式：

- `KV Cache Basic`：估算 decoder-only self-attention KV Cache 理论显存。
- `Deployment Sizing`：估算 GLM-5.2 在 vLLM / vLLM Ascend 推理部署时的 per-card 显存需求，并观察 TP/PP/DP/CP、MoE expert parallel、权重量化和 KV cache 量化的影响。

在线页面：

```text
https://hzchho.github.io/kv-cache-calculator/
```

## Quick Start

```powershell
cd kv-cache-calculator
python -m http.server 8000
```

然后访问：

```text
http://localhost:8000
```

## Basic KV Formula

```text
KV bytes = 2 * B * L * Hkv * T * d_head * bytes_per_elem
```

其中 `2` 表示 Key 和 Value 两份缓存。

## Deployment Formula

部署模式使用更接近 serving sizing 的 per-card 估算。对于 Ascend 上的 GLM-5.2-W4A8C8，默认把 `W4`、`A8`、`C8/KV cache` 分开配置：

```text
W4  -> weight bytes/param = 0.5
A8  -> activation bytes = 1
C8  -> KV bytes/element = 1
```

KV cache worst-case：

```text
logical_tokens = full_context_seqs * ceil(max_model_len / block_size) * block_size

effective_kv_tokens_per_card = ceil(logical_tokens / kv_cp_or_dcp_shard)

kv_per_card ~= effective_kv_tokens_per_card
                * ceil(L / PP)
                * ceil(Hkv / TP)
                * (k_dim + v_dim)
                * kv_bytes
```

权重显存估算：

```text
weight_per_card ~= model_weight_bytes / (TP * PP)
```

如果启用 MoE expert parallel：

```text
weight_per_card ~= non_expert_bytes / (TP * PP)
                  + expert_bytes / (TP * DP * PP)
```

per-card 预算：

```text
budget_per_card = card_memory_gib * memory_utilization
available_kv_per_card ~= budget_per_card - weight_per_card - runtime_reserve_gib
```

`runtime_reserve_gib` 用于预留 activation peak、graph、通信、fragmentation、临时 buffer 等静态公式难以精确建模的显存。

## GLM-5.2 Presets

默认预设包括：

- `GLM-5.2-W4A8C8 · Ascend`
  - target: Atlas 800 A3 64G × 16
  - `TP=16, DP=1, KV CP/DCP shard=16`
  - `max_model_len=1024000`
  - `full_context_seqs=1`
  - `vLLM max_num_seqs=32` 仅作为调度参数提示，不等价于 32 条 full 1M context 同时占满 KV。
- `GLM-5.2-W8A8 · Ascend`
- `GLM-5.2-FP8`
- `GLM-5.2-BF16`

所有模型结构、硬件、量化和并行参数都可以手动覆盖。

## Important Notes

这个工具是静态估算器，不替代 vLLM / vLLM Ascend 启动时的 profiler 和真实 benchmark。真实部署时请以日志中的 KV cache size、maximum concurrency、OOM 行为和压测结果为准。

GLM-5.2 属于 MoE + MLA/DSA 类架构，真实 KV cache layout 可能比标准 MHA/GQA 更复杂。当前工具默认使用 `qk_head_dim + v_head_dim` 做保守估算，并允许手动覆盖。

`full_context_seqs` 是用于 KV worst-case sizing 的“满长上下文序列数”；它和 vLLM 的 `max_num_seqs` 不是同一个概念。长上下文部署时，32 个 scheduler seqs 不意味着 32 个 1M context 同时常驻 KV。

## References

- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
- GLM-5.2-FP8 config: https://huggingface.co/zai-org/GLM-5.2-FP8/raw/main/config.json
- vLLM Ascend documentation: https://vllm-ascend.readthedocs.io/
- vLLM parallelism and scaling: https://docs.vllm.ai/en/v0.17.0/serving/parallelism_scaling/
- vLLM data parallel deployment: https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/
- vLLM CacheConfig: https://docs.vllm.ai/en/v0.14.1/api/vllm/config/cache/

## License

MIT License
