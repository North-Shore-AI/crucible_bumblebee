alias CrucibleBumblebee.{ExampleSurface, ForwardRunner}
alias CrucibleTap.TapPlan

tap_plan =
  TapPlan.new!(
    [
      [id: "hidden", signal_type: :middle_residuals, layers: [0]],
      [id: "logits", signal_type: :final_logits, layers: [:final]]
    ],
    plan_id: "forward-runner-mock"
  )

predict_fun = fn _inputs ->
  %{
    logits: Nx.tensor([[0.1, 0.4, 0.2]], type: :f32),
    hidden_states: {
      Nx.tensor([[1.0, 0.0]], type: :f32),
      Nx.tensor([[0.0, 1.0]], type: :f32),
      Nx.tensor([[1.0, 1.0]], type: :f32)
    },
    attentions: {Nx.tensor([[0.5, 0.5]], type: :f32)},
    cache: %{blocks: {:block0}}
  }
end

{:ok, trace} =
  ForwardRunner.run(predict_fun, %{}, tap_plan,
    trace_id: "trace-forward-runner-mock",
    model_id: "model:fixture",
    surface: ExampleSurface.surface(num_blocks: 1)
  )

IO.inspect(%{
  ok: true,
  example: "forward_runner_mock",
  signal_count: length(trace.signals),
  lifecycle: trace.metadata.lifecycle
})
