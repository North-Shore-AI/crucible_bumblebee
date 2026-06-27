<p align="center">
  <img src="assets/crucible_bumblebee.svg" width="200" height="200" alt="crucible_bumblebee logo" />
</p>

<p align="center">
  <a href="https://github.com/North-Shore-AI/crucible_bumblebee">
    <img alt="GitHub: crucible_bumblebee" src="https://img.shields.io/badge/GitHub-crucible_bumblebee-0b0f14?logo=github" />
  </a>
  <a href="https://github.com/North-Shore-AI/crucible_bumblebee/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0b0f14.svg" />
  </a>
</p>

# CrucibleBumblebee

Bumblebee, Axon, and Nx adapter layer for compiling Crucible tap plans into
model runs, hooks, traces, and decode steering.

## Stack Position

`crucible_bumblebee` is the implementation adapter. It knows Bumblebee model
surfaces, Axon hooks, Nx serving behavior, and decode-loop steering points.

## Installation

```elixir
def deps do
  [
    {:crucible_bumblebee, "~> 0.1.0"}
  ]
end
```

## Boundary

This package adapts Crucible contracts to Bumblebee/Nx/Axon. It does not own
Trinity orchestration, hosted runtime supervision, or the base signal ontology.

## Usage

```elixir
alias CrucibleBumblebee.{ExampleSurface, ForwardRunner}
alias CrucibleTap.TapPlan

tap_plan =
  TapPlan.new!([
    [id: "hidden", signal_type: :middle_residuals, layers: [12]],
    [id: "logits", signal_type: :final_logits, layers: [:final]]
  ])

surface = ExampleSurface.surface(num_blocks: 2)

{:ok, trace} =
  ForwardRunner.run(predict_fun, inputs, tap_plan,
    model_id: "model:local",
    surface: surface
  )
```

## Model Boundary

The reusable runners consume `CrucibleBumblebee.ModelSurface`. Qwen-family
support is provided as `Qwen3Surface`, an example surface module with its own
preflight artifact. Other model families provide their own surface module and
artifact; no runner assumes Qwen params paths or a 0.6B model.

## Mechanistic-Interpretability Metadata

The provider now tags emitted Bumblebee outputs with canonical activation
metadata:

- final logits: `unembed.hook_logits`
- hidden-state tuple entries: residual-stream names such as
  `blocks.0.hook_resid_pre`
- attention output collections: `blocks.N.attn.hook_pattern`
- fork-backed deep outputs: `blocks.N.attn.hook_q`, `hook_k`, `hook_v`,
  `hook_attn_scores`, `hook_z`, `blocks.N.hook_attn_out`,
  `blocks.N.mlp.hook_pre`, `hook_post`, `blocks.N.hook_mlp_out`, and
  layerwise residual streams

`CrucibleBumblebee.LogitLensRunner` projects raw in-memory hidden states through
surface-declared logit-lens parameter access. Summary-only traces do not
masquerade as raw activation caches; trace projection requires raw tensor refs
or an eager tensor resolver.

## Guides

- [Quickstart](guides/quickstart.md)
- [Concepts](guides/concepts.md)
- [Model Surfaces](guides/model_surfaces.md)
- [Preflight](guides/preflight.md)
- [Forward Runner](guides/forward_runner.md)
- [Interventions](guides/interventions.md)
- [Generation Runner](guides/generation_runner.md)
- [Backend](guides/backend.md)
- [Native Bumblebee Provider](guides/native_bumblebee_provider.md)
- [Generation Capability Degradation](guides/generation_capability_degradation.md)
- [Working Examples](guides/working_examples.md)
- [Testing](guides/testing.md)
- [Bumblebee Generation Surface](docs/bumblebee_generation_surface.md)

Documentation can be generated with `mix docs` and published to HexDocs.

## Status

Status: `native-model-backend-signal-generation-internals-passing`.

The native provider exercises real Hugging Face/Bumblebee models. The current
model ladder exercised `hf-internal-testing/tiny-random-gpt2`, `gpt2`,
`distilgpt2`, `hf-internal-testing/tiny-random-distilbert`, and
`trl-internal-testing/tiny-Qwen3ForCausalLM` where Bumblebee exposes a runnable
surface. Unsupported OPT and non-causal generation paths are recorded as
structured blockers.

The Binary backend ran locally. EXLA CPU/CUDA and Torchx were recorded as
unavailable on the local Elixir stack for this run. The signal/generation gates
captured input IDs, attention masks, final logits, top-k, entropy, margin,
backend events, generated tokens, and manual autoregressive generation-step
logits where the model was causal. Hidden states, attention collections, Q/K/V,
attention scores, attention outputs, MLP activations, residual streams, and
final norm telemetry are captured only when the compiled run requests and
receives the corresponding Bumblebee outputs.
Global intermediate logits, KV-cache metadata, and active mutation remain
structured surface blockers unless a provider advertises those exact
capabilities.

Artifacts are written under `tmp/crucible_v5/`, including:

```text
tmp/crucible_v5/reports/model_matrix.md
tmp/crucible_v5/reports/backend_matrix.md
tmp/crucible_v5/reports/signal_matrix.md
tmp/crucible_v5/reports/generation_matrix.md
tmp/crucible_v5/reports/internals_matrix.md
```
