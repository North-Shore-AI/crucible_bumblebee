# Concepts

`crucible_bumblebee` adapts Crucible tap, signal, trace, and policy contracts
to Bumblebee, Axon, and Nx. It does not own orchestration or hosted runtime
supervision.

The runner lifecycle is plan compilation, serving compilation, then execution.
Hook choices are fixed before serving execution unless a surface advertises
dynamic hook support.
