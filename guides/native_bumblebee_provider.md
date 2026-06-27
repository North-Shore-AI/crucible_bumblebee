# Native Bumblebee Provider

Purpose: run the native Elixir/Bumblebee provider against the real model
ladder.

## What this covers

This guide covers the CPU live forward gate for the model ladder. The gate
proves model/tokenizer loading, real tensor execution, final-logits extraction,
JSONL emission, matrix rows, and capability report emission where Bumblebee
exposes a runnable surface.

When compiled options cause Bumblebee to return hidden states or attentions,
`CrucibleBumblebee.SignalExtractor` records canonical activation metadata on the
resulting signals. It does not synthesize unavailable internals: Q/K/V and MLP
internals remain probe-only until a deeper instrumentation path emits those
tensors.

## Quickstart

```bash
CRUCIBLE_BUMBLEBEE_LIVE=true \
CRUCIBLE_BUMBLEBEE_MODEL_ID=gpt2 \
mix run examples/model_forward_live.exs -- --artifact-root tmp/crucible_v5
```

Expected output includes:

```elixir
%{
  ok: true,
  provider_kind: :elixir_bumblebee,
  model_id: "gpt2",
  forward_pass_ran?: true,
  trace_path: "tmp/crucible_v5/traces/native/model_forward_live.trace.jsonl"
}
```

## Artifacts

The forward gate writes:

```text
tmp/crucible_v5/traces/native/model_forward_live.trace.jsonl
tmp/crucible_v5/capability_reports/model_forward_live.capability_report.json
tmp/crucible_v5/reports/model_matrix.md
tmp/crucible_v5/reports/signal_matrix.md
```

When EXLA is unavailable, the runner uses the Binary backend and records the
selected backend in the trace.

## Activation Metadata

Canonical examples emitted by the native provider:

```elixir
%{activation_name: "unembed.hook_logits", axes: [:batch, :pos, :d_vocab]}
%{activation_name: "blocks.0.hook_resid_pre", axes: [:batch, :pos, :d_model]}
%{activation_name: "blocks.0.attn.hook_pattern", axes: [:batch, :head, :dest_pos, :src_pos]}
```

Required taps for `blocks.N.attn.hook_q`, `hook_k`, or `hook_v` fail closed until
deep Bumblebee instrumentation is enabled.

## Related guides

- [Generation Capability Degradation](generation_capability_degradation.md)
- [Backend](backend.md)
- [Testing](testing.md)
