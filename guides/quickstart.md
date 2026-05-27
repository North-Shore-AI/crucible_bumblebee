# Quickstart

Use a model surface, compile a tap plan, and run a prebuilt predict function.

```elixir
alias CrucibleBumblebee.{ExampleSurface, ForwardRunner}

surface = ExampleSurface.surface(num_blocks: 1)
{:ok, trace} = ForwardRunner.run(predict_fun, inputs, tap_plan, surface: surface)
```

Real model families should provide their own `ModelSurface` implementation and
surface preflight artifact.
