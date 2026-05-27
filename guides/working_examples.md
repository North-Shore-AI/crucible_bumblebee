# Working Examples

Run mock examples:

```bash
mix run examples/forward_runner_mock.exs
mix run examples/logit_steering_mock.exs
```

Live examples skip cleanly unless `CRUCIBLE_BUMBLEBEE_LIVE=1` is set:

```bash
mix run examples/model_forward_live.exs
mix run examples/model_generation_live.exs
```
