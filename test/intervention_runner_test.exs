defmodule CrucibleBumblebee.InterventionRunnerTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{ExampleSurface, InterventionRunner, ModelSurface}
  alias CrucibleMechInterp.{Intervention, SAE}

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

  test "SAE feature ablation interventions compile through active surfaces" do
    sae =
      SAE.from_map!(%{
        architecture: :standard,
        hook_name: "blocks.0.hook_resid_pre",
        apply_b_dec_to_input: false,
        params: %{
          w_enc: Nx.eye(1),
          w_dec: Nx.eye(1),
          b_enc: Nx.tensor([0.0]),
          b_dec: Nx.tensor([0.0])
        }
      })

    intervention =
      SAE.ablation_intervention(
        sae,
        "blocks.0.hook_resid_pre",
        Nx.tensor([[[2.0]]]),
        [0]
      )

    surface = active_surface()
    predict_fun = fn inputs -> %{logits: inputs.value} end

    executor = fn predict, inputs, [compiled], _model_surface, _opts ->
      {:ok,
       %{
         outputs: predict.(inputs),
         intervention_type: compiled.type,
         intervention_source: compiled.metadata.source,
         replacement: compiled.value
       }}
    end

    assert {:ok,
            %{
              outputs: %{logits: :patched},
              intervention_type: :replace,
              intervention_source: :sae_feature_ablation,
              replacement: replacement
            }} =
             InterventionRunner.run(predict_fun, %{value: :patched}, intervention, surface,
               executor: executor
             )

    assert Nx.to_flat_list(replacement) == [0.0]
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
