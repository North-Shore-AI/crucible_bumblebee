# Generation Runner

`GenerationRunner.generate/5` invokes an installed `Nx.Serving` with
`Nx.Serving.run/2` or an explicit custom loop. Telemetry-oriented cached decode
uses `CrucibleBumblebee.GenerationTrace`, which wraps
`Bumblebee.Text.Generation.build_generate/4` with `trace: true` and returns
per-step logits plus KV-cache offsets. Token-boundary steering requires
logits-processor capability or a custom loop. In-graph steering requires an
explicit `:in_graph_steering` capability.
