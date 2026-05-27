defmodule CrucibleBumblebee.GenerationRunnerMockTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{ExampleSurface, GenerationRunner, ModelSurface}
  alias CruciblePolicy.SteeringPlan

  test "generate wraps Nx.Serving.run for the installed serving surface" do
    serving = Nx.Serving.jit(&Nx.add(&1, 1))
    batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])

    assert {:ok, result} =
             GenerationRunner.generate(serving, batch, nil, ExampleSurface.surface(num_blocks: 1))

    assert Nx.to_flat_list(result) == [2, 3, 4]
  end

  test "token-boundary steering requires logits processor capability or custom loop" do
    surface =
      ModelSurface.new!(:minimal, [], %{
        surface_id: :minimal,
        capabilities: %{logits_processors: false}
      })

    steering = SteeringPlan.new!(trace_id: "trace-1", token_biases: %{1 => 1.0})

    assert {:error, :steering_surface_unavailable} =
             GenerationRunner.generate(
               Nx.Serving.jit(& &1),
               Nx.Batch.stack([Nx.tensor([1])]),
               steering,
               surface
             )

    custom_loop = fn input, plan -> %{input: input, steering_mode: plan.mode} end

    assert {:ok, %{steering_mode: :token_boundary}} =
             GenerationRunner.generate(custom_loop, "prompt", steering, surface,
               custom_loop: true
             )
  end
end
