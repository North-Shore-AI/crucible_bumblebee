# Model Surfaces

`ModelSurface` is the model-family boundary. A surface module declares an id,
family, capabilities, output options, preflight behavior, and optional
logit-lens access.

`ExampleSurface` is the generic fixture surface. `Qwen3Surface` is an optional
Qwen-family example. Runners consume the behaviour and do not hard-code either
surface.

Surface nodes may include canonical activation metadata. A node should advertise
`:read` only when the provider can emit a real signal for that node in the
current implementation. Deep internals that are known by name but not yet
capturable should remain `[:probe]` so required read taps fail closed during tap
compilation.
