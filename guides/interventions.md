# Interventions

`CrucibleBumblebee.InterventionRunner` compiles
`CrucibleMechInterp.Intervention` values against a `ModelSurface`.

Bundled Bumblebee surfaces currently expose read-only activation internals.
They do not advertise `:fuse` or `:gate`, so replace/add/source-swap/zero
interventions fail closed before model execution.

```elixir
intervention =
  CrucibleMechInterp.Intervention.replace(
    "blocks.0.hook_resid_pre",
    replacement_tensor
  )

{:error, {:unsupported_intervention, _, _}} =
  CrucibleBumblebee.InterventionRunner.compile(
    intervention,
    CrucibleBumblebee.ExampleSurface.surface(num_blocks: 1)
  )
```

When a provider surface does advertise active operations, callers must pass a
real executor callback to `run/5`. The executor owns graph rewrite or equivalent
model rerun behavior and receives the predict function, inputs, compiled
interventions, surface, and options. The runner does not synthesize patched
outputs.
