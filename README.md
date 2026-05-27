<p align="center">
  <img src="assets/crucible_bumblebee.svg" width="200" height="200" alt="crucible_bumblebee logo" />
</p>

<p align="center">
  <a href="https://github.com/North-Shore-AI/crucible_bumblebee">
    <img alt="GitHub: crucible_bumblebee" src="https://img.shields.io/badge/GitHub-crucible_bumblebee-0b0f14?logo=github" />
  </a>
  <a href="https://github.com/North-Shore-AI/crucible_bumblebee/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0b0f14.svg" />
  </a>
</p>

# CrucibleBumblebee

Bumblebee, Axon, and Nx adapter layer for compiling Crucible tap plans into
model runs, hooks, traces, and decode steering.

## Stack Position

`crucible_bumblebee` is the implementation adapter. It knows Bumblebee model
surfaces, Axon hooks, Nx serving behavior, and decode-loop steering points.

## Installation

```elixir
def deps do
  [
    {:crucible_bumblebee, "~> 0.1.0"}
  ]
end
```

## Boundary

This package adapts Crucible contracts to Bumblebee/Nx/Axon. It does not own
Trinity orchestration, hosted runtime supervision, or the base signal ontology.

Documentation can be generated with `mix docs` and published to HexDocs.
