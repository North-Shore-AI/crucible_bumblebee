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
    model_ref: "model:local",
    surface: surface
  )
```

## Model Boundary

The reusable runners consume `CrucibleBumblebee.ModelSurface`. Qwen-family
support is provided as `Qwen3Surface`, an example surface module with its own
preflight artifact. Other model families provide their own surface module and
artifact; no runner assumes Qwen params paths or a 0.6B model.

## Guides

- [Quickstart](guides/quickstart.md)
- [Concepts](guides/concepts.md)
- [Model Surfaces](guides/model_surfaces.md)
- [Preflight](guides/preflight.md)
- [Forward Runner](guides/forward_runner.md)
- [Generation Runner](guides/generation_runner.md)
- [Backend](guides/backend.md)
- [Native Bumblebee Provider](guides/native_bumblebee_provider.md)
- [Generation Capability Degradation](guides/generation_capability_degradation.md)
- [Working Examples](guides/working_examples.md)
- [Testing](guides/testing.md)
- [Bumblebee Generation Surface](docs/bumblebee_generation_surface.md)

Documentation can be generated with `mix docs` and published to HexDocs.

## V5 Status

Status: `native-model-backend-signal-generation-internals-passing`.

V5 expands the native provider beyond the initial tiny-GPT2 control path. The
model ladder exercised `hf-internal-testing/tiny-random-gpt2`, `gpt2`,
`distilgpt2`, `hf-internal-testing/tiny-random-distilbert`, and
`trl-internal-testing/tiny-Qwen3ForCausalLM` where Bumblebee exposes a runnable
surface. Unsupported OPT and non-causal generation paths are recorded as
structured blockers.

The Binary backend ran locally. EXLA CPU/CUDA and Torchx were recorded as
unavailable on the local Elixir stack for this run. The signal/generation gates
captured input IDs, attention masks, final logits, top-k, entropy, margin,
backend events, generated tokens, and manual autoregressive generation-step
logits where the model was causal. Hidden states, attentions, global
intermediate logits, KV-cache metadata, and active mutation remain structured
Bumblebee/Axon surface blockers unless a provider advertises those capabilities.

V5 artifacts are written under `tmp/crucible_v5/`, including:

```text
tmp/crucible_v5/reports/model_matrix.md
tmp/crucible_v5/reports/backend_matrix.md
tmp/crucible_v5/reports/signal_matrix.md
tmp/crucible_v5/reports/generation_matrix.md
tmp/crucible_v5/reports/internals_matrix.md
```
