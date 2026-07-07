# EMLX Qwen3 Bridge

`CrucibleBumblebee.EMLXQwen3` is the Crucible-facing bridge for the native
EMLX Qwen3 trace implementation.

The bridge is intentionally thin:

- EMLX owns native execution, lazy MLX tensors, Qwen3 forward tracing, cached
  generation tracing, and residual interventions.
- Crucible owns surface metadata, capability reporting, public redaction, and
  conversion of EMLX trace maps into `CrucibleMechInterp.ActivationCache`.
- `crucible_bumblebee` does not compile against `emlx_axon` by default, so
  normal CI does not require EMLX NIF loading or Apple Silicon hardware.

## Dependency Pin

Use the EMLX fork branch recorded by the bridge:

```elixir
CrucibleBumblebee.EMLXQwen3.dependency_pin()
```

Current pin:

- repo: `https://github.com/North-Shore-AI/emlx.git`
- branch: `phase-9-qwen3-trace`
- ref: `16629bc936c6e1919fc97232e8615d48a262885d`
- sparse package: `emlx_axon`

The fork branch is kept up to date with `elixir-nx/emlx` main (merged
through the fused kv_cache+sdpa PR, elixir-nx/emlx#124) so the trace
implementation benefits from upstream's native compiler and NIF
improvements. Upstream also shipped its own, independent
`EMLXAxon.Qwen3.{Model,Generate}` under the same module names (a
non-instrumented dense-generation path with no tracing support) — the fork
keeps our own instrumented implementation and does not adopt upstream's
competing one, since this bridge depends on `generate_trace/3`, which
upstream's version never had.

## Surface

```elixir
surface = CrucibleBumblebee.EMLXQwen3.surface(num_blocks: 28)
```

The returned `CrucibleBumblebee.ModelSurface` uses adapter `:emlx` and Qwen3
canonical activation names:

- `blocks.N.hook_resid_pre`
- `blocks.N.hook_resid_mid`
- `blocks.N.hook_resid_post`
- `blocks.N.attn.hook_q`
- `blocks.N.attn.hook_k`
- `blocks.N.attn.hook_v`
- `blocks.N.attn.hook_scores`
- `blocks.N.attn.hook_pattern`
- `blocks.N.attn.hook_z`
- `blocks.N.mlp.hook_pre`
- `blocks.N.mlp.hook_post`
- `blocks.N.hook_mlp_out`
- `ln_final.hook_scale`
- `ln_final.hook_normalized`
- `unembed.hook_logits`

## Capabilities

```elixir
capabilities = CrucibleBumblebee.EMLXQwen3.capabilities(num_blocks: 28)
```

The report includes:

- final logits
- KV-cache metadata
- cached generation trace
- residual-stream interventions
- attention Q/K/V, scores, pattern, and Z captures
- MLP activations
- norm telemetry
- lazy tensor refs
- canonical activation claims

Head-ablation intervention is explicitly unsupported in this phase.

## Generation Trace Conversion

Run EMLX generation with `EMLXAxon.Qwen3.Generate.generate_trace/3`, then
normalize the returned map:

```elixir
{tokens, %{trace: trace}} =
  EMLXAxon.Qwen3.Generate.generate_trace(input_ids, state,
    max_new_tokens: 32,
    sampler: :greedy,
    capture: [:cache_metadata],
    trace_logits: true
  )

{:ok, normalized} =
  CrucibleBumblebee.EMLXQwen3.normalize_generation_trace({tokens, %{trace: trace}})
```

For public reports, redact raw tensor refs:

```elixir
public_step =
  normalized.steps
  |> hd()
  |> CrucibleBumblebee.EMLXQwen3.public_step()
```

To analyze generation logits as a cache:

```elixir
{:ok, cache} = CrucibleBumblebee.EMLXQwen3.to_activation_cache(normalized)
```

The cache stores per-step logits at `unembed.hook_logits` with axes
`[:batch, :pos, :d_vocab]`.
