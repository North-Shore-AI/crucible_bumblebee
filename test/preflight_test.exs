defmodule CrucibleBumblebee.PreflightTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{ModelBundle, Preflight}
  alias CrucibleBumblebee.Surfaces.Gpt2CausalLMSurface

  test "selects a generic GPT-2 surface for any GPT-2-family bundle" do
    bundle = bundle(model_id: "gpt2", model_family: :gpt2)

    assert {:ok, surface} = Preflight.surface_for_bundle(bundle)
    assert surface.id == Gpt2CausalLMSurface.id()
    assert surface.family == :gpt2
  end

  test "does not reject distilgpt2 as an unexpected model id" do
    bundle = bundle(model_id: "distilgpt2", model_family: :gpt2)

    assert {:ok, preflight} = Preflight.run(bundle, CrucibleBumblebee.Live.forward_tap_plan())
    assert preflight.capability_report.supported == ["final_logits"]
    assert preflight.capability_report.optional_dropped == ["hidden_state", "attention_weights"]
  end

  test "returns a structured unsupported surface family for encoder-only families" do
    bundle = bundle(model_id: "hf-internal-testing/tiny-random-BertModel", model_family: :bert)

    assert {:error, {:unsupported_live_surface_family, :bert}} =
             Preflight.run(bundle, CrucibleBumblebee.Live.forward_tap_plan())
  end

  test "selects final-logits surface for sequence classification families" do
    bundle =
      bundle(model_id: "hf-internal-testing/tiny-random-distilbert", model_family: :distilbert)

    assert {:ok, preflight} = Preflight.run(bundle, CrucibleBumblebee.Live.forward_tap_plan())
    assert preflight.surface.id == :distilbert_final_logits
    assert preflight.capability_report.model_family == :distilbert
    assert preflight.capability_report.supported == ["final_logits"]
  end

  defp bundle(attrs) do
    attrs = Map.new(attrs)

    struct!(
      ModelBundle,
      Map.merge(
        %{
          model_id: "gpt2",
          tokenizer_id: "gpt2",
          tokenizer: :tokenizer,
          params: %{},
          backend: :binary,
          model_family: :gpt2,
          revision: nil,
          spec: nil
        },
        attrs
      )
    )
  end
end
