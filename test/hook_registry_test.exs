defmodule CrucibleBumblebee.HookRegistryTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{ExampleSurface, HookRegistry}

  test "lists Axon graph nodes with hook candidate signals" do
    model =
      Axon.input("input", shape: {nil, 4})
      |> Axon.dense(3)
      |> Axon.dense(2)

    nodes = HookRegistry.list_nodes(model)

    assert Enum.any?(nodes, &(&1.op_name == :dense))
    assert Enum.any?(nodes, &(:mlp_activation in &1.candidate_signals))

    candidates = HookRegistry.candidate_taps(nodes)
    assert Enum.find(candidates, &(&1.target_signal == :final_logits))
    assert Enum.find(candidates, &(&1.target_signal == :mlp_activation))
  end

  test "resolves Crucible surface taps through portable selectors" do
    surface = ExampleSurface.surface(num_blocks: 2)

    assert {:ok, [node]} =
             HookRegistry.resolve_tap(surface,
               signal_type: :middle_residuals,
               layer: :last
             )

    assert node.layer_index == 1
    assert node.layer_name == "decoder.layers.1.mlp.output"
  end

  test "missing tap resolution is explicit" do
    surface = ExampleSurface.surface(num_blocks: 1)

    assert {:error, {:tap_not_found, _selector}} =
             HookRegistry.resolve_tap(surface, signal_type: :router_logits)
  end
end
