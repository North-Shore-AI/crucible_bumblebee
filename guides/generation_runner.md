# Generation Runner

`GenerationRunner.generate/5` invokes an installed `Nx.Serving` with
`Nx.Serving.run/2` or an explicit custom loop. Token-boundary steering requires
logits-processor capability or a custom loop. In-graph steering requires an
explicit `:in_graph_steering` capability.
