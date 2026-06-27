# Generation Capability Degradation

Purpose: explain generation behavior and capability degradation.

## What this covers

`model_generation_live.exs` loads the configured real model profile. It first
optionally attempts the high-level Bumblebee generation surface for decoded text,
then uses `CrucibleBumblebee.GenerationTrace` for the authoritative telemetry
path. The trace path runs Bumblebee's cached greedy decode and returns selected
token IDs, per-step processed logits, and KV-cache offsets. If a signal still
cannot be captured, the runner writes an explicit degraded capability report
instead of pretending the signal exists.

## Quickstart

```bash
CRUCIBLE_BUMBLEBEE_LIVE=true \
CRUCIBLE_BUMBLEBEE_MODEL_ID=gpt2 \
mix run examples/model_generation_live.exs -- --artifact-root tmp/crucible_v5
```

Expected output:

```elixir
%{
  ok: true,
  generation_supported?: true,
  generation_success_level: :kv_cache_generation_trace
}
```

The trace contains `token_step`, `generation_step`, and
`generation_step_logits` records when the cached decode trace succeeds. Each
step includes bounded KV-cache metadata, currently cache offset, prompt length,
generated length, and max length. Unsupported models or non-greedy trace
requests are recorded as structured blockers.

## Related guides

- [Native Bumblebee Provider](native_bumblebee_provider.md)
- [Generation Trace](generation_trace.md)
- [Generation Runner](generation_runner.md)
