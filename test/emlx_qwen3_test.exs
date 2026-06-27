defmodule CrucibleBumblebee.EMLXQwen3Test do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{EMLXQwen3, ModelSurface}
  alias CrucibleMechInterp.ActivationCache

  test "surface declares the EMLX adapter and pinned dependency" do
    surface = EMLXQwen3.surface(num_blocks: 1)

    assert %ModelSurface{id: :emlx_qwen3, family: :qwen3} = surface
    assert surface.surface.adapter == :emlx
    assert surface.metadata.emlx_dependency.ref == EMLXQwen3.dependency_pin().ref
    assert surface.capabilities.generation_trace
    assert surface.capabilities.residual_interventions
    assert surface.capabilities.interventions.residual
    refute surface.capabilities.interventions.head_ablation
    assert Map.has_key?(surface.capabilities.activations, "blocks.0.attn.hook_pattern")

    assert Enum.any?(surface.surface.nodes, fn node ->
             node.activation_name == "blocks.0.attn.hook_pattern"
           end)
  end

  test "exports registry-compatible model internals metadata" do
    compatibility = EMLXQwen3.provider_compatibility(num_blocks: 1)

    assert compatibility.provider_kind == :emlx_qwen3
    assert compatibility.model_family == :qwen3
    assert :attention_qkv in compatibility.supported_capture_groups
    assert :kv_cache_generation_trace in compatibility.supported_generation_features
    assert :residual_intervention in compatibility.supported_active_controls
    assert :head_ablation in compatibility.unsupported_active_controls
    assert "blocks.0.attn.hook_q" in compatibility.supported_activations
    assert "unembed.hook_logits" in compatibility.supported_activations
  end

  test "normalizes EMLX generation traces and redacts public steps" do
    trace = tiny_trace()

    assert {:ok, normalized} = EMLXQwen3.normalize_generation_trace(trace)
    assert normalized.provider == :emlx_qwen3
    assert normalized.generation_success_level == :emlx_qwen3_generation_trace
    assert normalized.generated_token_ids == [4, 5]
    assert normalized.trace_metadata.emitted_steps == 2

    public = trace.steps |> hd() |> EMLXQwen3.public_step()
    refute Map.has_key?(public, :logits)
    refute Map.has_key?(public, :model_trace)
    refute Map.has_key?(public.cache_metadata, :layers)
    assert public.cache_metadata.current_length == 3
  end

  test "generation logits can be exposed as an activation cache" do
    trace = tiny_trace()

    assert {:ok, cache} = EMLXQwen3.to_activation_cache(trace)
    assert Nx.shape(ActivationCache.get!(cache, "unembed.hook_logits")) == {1, 2, 3}
    assert cache.metadata.source == :emlx_qwen3_generation_trace
    assert cache.metadata.generated_token_ids == [4, 5]
    assert cache.specs["unembed.hook_logits"].signal_type == :generation_step_logits
  end

  defp tiny_trace do
    %{
      generated_token_ids: [4, 5],
      prompt_length: 2,
      requested_max_new_tokens: 2,
      steps: [
        %{
          step_index: 1,
          phase: :prefill,
          token_id: 4,
          logits: Nx.tensor([[0.1, 0.2, 0.7]]),
          logits_shape: {1, 3},
          cache_metadata: %{
            current_length: 3,
            layers: [%{layer_index: 0, key_shape: {1, 1, 8, 2}}]
          },
          model_trace: %{activations: %{}}
        },
        %{
          step_index: 2,
          phase: :decode,
          token_id: 5,
          logits: Nx.tensor([[0.3, 0.4, 0.3]]),
          logits_shape: {1, 3},
          cache_metadata: %{current_length: 4},
          model_trace: %{activations: %{}}
        }
      ]
    }
  end
end
