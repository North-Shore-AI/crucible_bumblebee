alias CrucibleBumblebee.LogitsProcessor
alias CruciblePolicy.SteeringPlan

plan =
  SteeringPlan.new!(
    trace_id: "trace-logit-steering",
    token_biases: %{1 => 1.0},
    energies: [%{source: :policy_rule, kind: :safety, energy: %{2 => 0.5}, weight: 2.0}],
    banned_token_ids: [0]
  )

IO.inspect(%{
  ok: true,
  example: "logit_steering_mock",
  logits: LogitsProcessor.process([1.0, 1.0, 1.0], plan)
})
