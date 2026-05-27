# Generation Capability Degradation

Purpose: explain V4 generation behavior when step-level logits are unavailable.

## What this covers

`model_generation_live.exs` loads the same real tiny GPT-2 profile as the
forward gate. V4 does not claim native step-logit hooks when Bumblebee's
generation surface hides those internals, so it writes an explicit degraded
capability report instead of skipping.

## Quickstart

```bash
CRUCIBLE_BUMBLEBEE_LIVE=true mix run examples/model_generation_live.exs
```

Expected output:

```elixir
%{
  ok: true,
  generation_supported?: false,
  reason: :generation_logits_unavailable
}
```

The trace contains `generation_start`, `generation_end` with
`status: "degraded"`, and `trace_end`.

## Related guides

- [Native Bumblebee Provider](native_bumblebee_provider.md)
- [Generation Runner](generation_runner.md)
