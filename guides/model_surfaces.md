# Model Surfaces

`ModelSurface` is the model-family boundary. A surface module declares an id,
family, capabilities, output options, preflight behavior, and optional
logit-lens access.

`ExampleSurface` is the generic fixture surface. `Qwen3Surface` is an optional
Qwen-family example. Runners consume the behaviour and do not hard-code either
surface.
