# Generation Capability Degradation

Purpose: explain generation behavior and capability degradation.

## What this covers

`model_generation_live.exs` loads the configured real model profile. It first
attempts the high-level Bumblebee generation surface, then uses a manual
autoregressive forward loop when that surface hides step logits. If a signal
still cannot be captured, the runner writes an explicit degraded capability
report instead of pretending the signal exists.

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
  generation_success_level: :generation_step_logits
}
```

The trace contains generation events and `generation_step_logits` records when
the manual loop succeeds. Unsupported models or hidden KV-cache surfaces are
recorded as structured blockers.

## Related guides

- [Native Bumblebee Provider](native_bumblebee_provider.md)
- [Generation Runner](generation_runner.md)
