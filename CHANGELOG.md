# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added the bounded V4 native tiny GPT-2 runner, model bundle/loader, curated
  `TinyGPT2Surface`, preflight, trace writer, and live forward/generation
  scripts writing `tmp/crucible_v4/*.jsonl`.
- Added explicit generation step-logit degradation reports.
- Added model-agnostic `ModelSurface` behaviour, generic example surface, and optional Qwen-family example surface.
- Added versioned surface preflight artifacts, generic logit-lens access, backend diagnostics, and serving lifecycle metadata.
- Added `GenerationRunner.generate/5`, examples, guides, and live-test tags that skip by default.
