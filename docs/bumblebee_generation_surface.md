# Bumblebee Generation Surface

Pinned dependency surface:

- Bumblebee source/ref: `elixir-nx/bumblebee@d0774e8ab8c4d5ac60ade95ec8dc9e1f0efd7306`
- Nx lock version: `0.12.1`
- Axon lock version: `0.8.1`

The installed Bumblebee text-generation path builds an `Nx.Serving` through
the hidden `generation/4` function in the `TextGeneration` module under
`Bumblebee.Text`. Internally, that function calls
`Bumblebee.Text.Generation.build_generate/4`, which accepts
`:logits_processors` and returns a numerical generation function. Invocation is
through `Nx.Serving.run/2`, or through a streaming `Nx.Serving` when the caller
builds generation with `stream: true`.

`GenerationRunner.generate/5` therefore wraps a prebuilt `Nx.Serving` or an
explicit custom loop. It does not assume a nonexistent helper name. Steering is
accepted only when the surface advertises logits-processor support, in-graph
steering support, or the caller explicitly selects a custom loop.
