# Generation Trace

`CrucibleBumblebee.GenerationTrace` is the KV-cache-aware decode telemetry path.
It calls `Bumblebee.Text.Generation.build_generate/4` with `trace: true`, so
token selection, cache updates, logits processing, and EOS handling stay inside
Bumblebee's real generation implementation.

## Captured Fields

Each step contains:

- `step_index` and generated-token index
- selected token ID and decoded token text when a tokenizer is available
- processed step logits as an in-memory tensor
- bounded tensor summary with entropy, top-k, and margin
- KV-cache offset from Bumblebee's decode cache
- bounded cache metadata: prompt length, generated length, max length, batch
  index, and trace source

Raw cache tensors are not written to JSONL. Trace events write bounded metadata
and signal records write tensor summaries.

## Activation Cache

Use `GenerationTrace.to_activation_cache/2` when generation-step logits need to
enter the TransformerLens-style cache API:

```elixir
{:ok, cache} = CrucibleBumblebee.GenerationTrace.to_activation_cache(trace)
logits = CrucibleMechInterp.ActivationCache.get!(cache, "unembed.hook_logits")
```

The cache stores `unembed.hook_logits` with axes `[:batch, :pos, :d_vocab]`,
where `:pos` is the generated-token index. This cache represents generated-step
logits only; it does not claim residual, hidden-state, attention, or MLP
captures. If those internals are requested from the generation trace path,
`optional_internals` returns an explicit unsupported reason until Bumblebee
exposes those tensors inside cached generation.

## Supported Path

The current trace path supports single-item greedy causal-language-model
generation. Non-greedy strategies fail closed with
`{:unsupported_generation_trace_strategy, strategy}`. Batched traces fail closed
with `{:batch_generation_trace_not_yet_supported, batch_size}` until the public
result shape grows a batch-preserving return contract.

## Example

```elixir
{:ok, trace} =
  CrucibleBumblebee.GenerationTrace.run(bundle, "Hi",
    max_new_tokens: 4,
    top_k: 10,
    seed: 0
  )

Enum.map(trace.steps, &{&1.token_id, &1.cache_offset})
```

`CrucibleBumblebee.Live.generation/1` uses this module for live artifacts. The
trace stream includes `token_step`, `generation_step`, and
`generation_step_logits` signal records with cache metadata attached.
