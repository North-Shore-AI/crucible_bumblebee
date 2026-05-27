defmodule CrucibleBumblebee.Surfaces.Gpt2CausalLMSurfaceTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.Surfaces.Gpt2CausalLMSurface

  test "advertises only the currently proven GPT-2 final logits surface" do
    surface = Gpt2CausalLMSurface.surface()

    assert surface.id == :gpt2_causal_lm
    assert surface.family == :gpt2
    assert surface.capabilities.final_logits
    refute surface.capabilities.hidden_state
    refute surface.capabilities.attention_weights
  end

  test "preflight accepts GPT-2 family metadata and rejects mismatched families" do
    assert {:ok, preflight} = Gpt2CausalLMSurface.preflight(%{model_family: :gpt2}, [])
    assert preflight.post_processing_extractors == [:final_logits]

    assert {:error, {:surface_family_mismatch, :bert}} =
             Gpt2CausalLMSurface.preflight(%{model_family: :bert}, [])
  end
end
