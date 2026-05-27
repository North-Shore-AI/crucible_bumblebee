# Forward Runner

`ForwardRunner.compile_serving/3` compiles a tap plan against a surface and
returns a Crucible serving contract. `ForwardRunner.run/4` and
`ForwardRunner.run_serving/3` then execute the predict function and convert
outputs into a `ForwardTrace`.

Per-request changes should stay in post-processing unless the model surface
advertises dynamic hooks.
