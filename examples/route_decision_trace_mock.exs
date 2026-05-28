alias CrucibleBumblebee.{ExampleSurface, ForwardRunner}
alias CrucibleSignalTrace.JSONL
alias CrucibleTap.TapPlan

tap_plan =
  TapPlan.new!(
    [
      [id: "hidden", signal_type: :middle_residuals, layers: [0]],
      [id: "logits", signal_type: :final_logits, layers: [:final]]
    ],
    plan_id: "route-decision-trace-mock"
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
    trace_id: "route-decision-trace",
    model_id: "model:fixture",
    surface: ExampleSurface.surface(num_blocks: 1)
  )

{:ok, decision} = CruciblePolicy.decide(trace)

path = Path.join(System.tmp_dir!(), "crucible_bumblebee_route_decision_trace.jsonl")
File.rm(path)

:ok = JSONL.append(path, trace)
:ok = JSONL.append(path, decision)

IO.puts(
  Jason.encode!(%{
    trace_id: trace.trace_id,
    signal_count: length(trace.signals),
    selected_target: decision.selected_target,
    jsonl_path: path
  })
)
