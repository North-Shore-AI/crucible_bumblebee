# Backend

`Backend.prefer/1` returns structured diagnostics:

- `{:ok, backend}` for selected backends
- `{:error, {:not_installed, dep}}` when EXLA or Torchx is missing
- `{:error, {:no_device, backend, detail}}` when a backend cannot initialize
- `{:error, {:no_available_backend, attempts}}` when `:auto` exhausts options

`Backend.reset/0` restores only the process-local backend captured by
`prefer/1`.
