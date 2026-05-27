# Preflight

Surface preflight records public model-surface bindings before hooks are
compiled. Artifacts live under `priv/surface_preflight` and include the
surface id, module, dependency fingerprint, nodes, extractors, unsupported
features, and logit-lens access paths when available.

If the dependency fingerprint changes, regenerate the artifact before relying
on named hooks.
