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

The bundled `ExampleSurface` and `Qwen3Surface` advertise exact reads for the
deep outputs exposed by the pinned North-Shore-AI Bumblebee fork:

- attention Q/K/V, attention scores, attention pattern, per-head `hook_z`, and
  projected `hook_attn_out`
- MLP gate/pre, post, and output activations
- layerwise residual stream pre, mid, and post outputs

Surface modules for other model families should only copy those `:read`
capabilities after their forward path returns the corresponding tensors.
