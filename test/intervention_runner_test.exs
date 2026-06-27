defmodule CrucibleBumblebee.InterventionRunnerTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{ExampleSurface, InterventionRunner, ModelSurface}
  alias CrucibleMechInterp.Intervention

  test "bundled read-only surfaces fail closed for active interventions" do
    intervention = Intervention.replace("blocks.0.hook_resid_pre", Nx.tensor([[[0.0]]]))

    assert {:error, {:unsupported_intervention, "blocks.0.hook_resid_pre", :replace}} =
             InterventionRunner.compile(intervention, ExampleSurface.surface(num_blocks: 1))
  end

  test "active-capable surfaces require a real executor callback" do
    surface = active_surface()
    intervention = Intervention.replace("blocks.0.hook_resid_pre", Nx.tensor([[[1.0]]]))
    predict_fun = fn inputs -> %{logits: inputs.value} end

    assert {:error, :intervention_executor_required} =
             InterventionRunner.run(predict_fun, %{value: :unchanged}, intervention, surface)

    executor = fn predict, inputs, interventions, model_surface, _opts ->
      {:ok,
       %{
         outputs: predict.(inputs),
         intervention_count: length(interventions),
         surface_id: model_surface.id
       }}
    end

    assert {:ok,
            %{
              outputs: %{logits: :patched},
              intervention_count: 1,
              surface_id: :active_fixture
            }} =
             InterventionRunner.run(predict_fun, %{value: :patched}, intervention, surface,
               executor: executor
             )
  end

  defp active_surface do
    ModelSurface.new!(
      :active_fixture,
      [
        [
          id: "resid-pre",
          signal_type: :residual_stream,
          activation_name: "blocks.0.hook_resid_pre",
          layer_index: 0,
          operations: [:read, :fuse],
          capture_modes: [:summary]
        ]
      ],
      %{surface_id: :active_fixture}
    )
  end
end
