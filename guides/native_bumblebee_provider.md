# Native Bumblebee Provider

Purpose: run the bounded V4 native Elixir/Bumblebee provider against tiny GPT-2.

## What this covers

This guide covers the CPU live forward gate for
`hf-internal-testing/tiny-random-gpt2`. The gate proves model/tokenizer loading,
real tensor execution, final-logits extraction, v4 JSONL emission, and capability
report emission.

## Quickstart

```bash
CRUCIBLE_BUMBLEBEE_LIVE=true mix run examples/model_forward_live.exs
```

Expected output includes:

```elixir
%{
  ok: true,
  provider_kind: :elixir_bumblebee,
  model_id: "hf-internal-testing/tiny-random-gpt2",
  forward_pass_ran?: true,
  trace_path: "tmp/crucible_v4/model_forward_live.trace.jsonl"
}
```

## Artifacts

The forward gate writes:

```text
tmp/crucible_v4/model_forward_live.trace.jsonl
tmp/crucible_v4/model_forward_live.capability_report.json
```

When EXLA is unavailable, the runner uses the Binary backend and records the
selected backend in the trace.

## Related guides

- [Generation Capability Degradation](generation_capability_degradation.md)
- [Backend](backend.md)
- [Testing](testing.md)
