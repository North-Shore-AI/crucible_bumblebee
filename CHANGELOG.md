# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added the initial native tiny GPT-2 runner, model bundle/loader, curated
  `TinyGPT2Surface`, preflight, trace writer, and live forward/generation
  scripts.
- Added explicit generation step-logit degradation reports.
- Added V5 model, backend, signal, generation, and internals ladders over real
  Hugging Face/Bumblebee models with structured blockers for unsupported
  families, unavailable backends, hidden states, attentions, KV-cache metadata,
  and active mutation.
- Added manual autoregressive generation-step-logit capture for supported
  causal models and V5 matrix reports under `tmp/crucible_v5/reports`.
- Added model-agnostic `ModelSurface` behaviour, generic example surface, and optional Qwen-family example surface.
- Added versioned surface preflight artifacts, generic logit-lens access, backend diagnostics, and serving lifecycle metadata.
- Added `GenerationRunner.generate/5`, examples, guides, and live-test tags that skip by default.
