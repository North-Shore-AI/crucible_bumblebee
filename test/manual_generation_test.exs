defmodule CrucibleBumblebee.ManualGenerationTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.ManualGeneration

  test "greedy token id selects the highest final-axis logit" do
    logits = Nx.tensor([[[0.1, 0.7, 0.2]]])

    assert ManualGeneration.greedy_token_id(logits) == 1
  end

  test "append token extends input ids and attention mask" do
    inputs = %{
      "input_ids" => Nx.tensor([[40, 73]], type: {:u, 32}),
      "attention_mask" => Nx.tensor([[1, 1]], type: {:u, 32})
    }

    updated = ManualGeneration.append_token(inputs, 9)

    assert Nx.to_flat_list(updated["input_ids"]) == [40, 73, 9]
    assert Nx.to_flat_list(updated["attention_mask"]) == [1, 1, 1]
  end

  test "public step drops raw logits and keeps bounded summary" do
    step = %{
      step_index: 1,
      token_id: 7,
      logits: Nx.tensor([1, 2, 3]),
      tensor_summary: Crucible.TensorSummary.compute(Nx.tensor([1, 2, 3]))
    }

    public = ManualGeneration.public_step(step)

    refute Map.has_key?(public, :logits)
    assert public.tensor_summary.rank == 1
  end
end
