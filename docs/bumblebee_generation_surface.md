# Bumblebee Generation Surface

Pinned dependency surface:

- Bumblebee source/ref: `North-Shore-AI/bumblebee@cbe271afafcacff04d298046f4b11711712b4123`
- Nx lock version: `0.12.1`
- Axon lock version: `0.8.1`

The installed Bumblebee text-generation path builds an `Nx.Serving` through
the hidden `generation/4` function in the `TextGeneration` module under
`Bumblebee.Text`. Internally, that function calls
`Bumblebee.Text.Generation.build_generate/4`, which accepts
`:logits_processors`, `:ignore_output`, and the fork-backed `trace: true`
option. Invocation is through `Nx.Serving.run/2`, or through a streaming
`Nx.Serving` when the caller builds generation with `stream: true`.

`GenerationRunner.generate/5` therefore wraps a prebuilt `Nx.Serving` or an
explicit custom loop. `CrucibleBumblebee.GenerationTrace` uses the numerical
builder directly with `trace: true` and emits selected token IDs, per-step
processed logits, and per-step cache offsets. Steering is accepted only when the
surface advertises logits-processor support, in-graph steering support, or the
caller explicitly selects a custom loop.
