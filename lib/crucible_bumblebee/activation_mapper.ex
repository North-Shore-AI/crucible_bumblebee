defmodule CrucibleBumblebee.ActivationMapper do
  @moduledoc """
  Maps Bumblebee output keys and surface node names to canonical activation names.

  This module only describes activations that are present in current Bumblebee
  outputs or declared provider surfaces. Exact deep Q/K/V capture is still a
  separate instrumentation phase.
  """

  alias CrucibleSignal.ActivationMetadata

  @doc "Metadata for final logits emitted by causal language model heads."
  def final_logits do
    metadata("unembed.hook_logits",
      source_output: :logits,
      capture_exactness: :exact_output,
      capture_mode: :summary
    )
  end

  @doc "Metadata for one hidden-state tuple entry from Bumblebee outputs."
  def hidden_state(index, size) when is_integer(index) and is_integer(size) and size > 0 do
    n_layers = max(size - 1, 1)

    name =
      cond do
        index >= n_layers -> "blocks.#{n_layers - 1}.hook_resid_post"
        true -> "blocks.#{index}.hook_resid_pre"
      end

    metadata(name,
      source_output: :hidden_states,
      capture_exactness: :bumblebee_output_hidden_state,
      capture_mode: :summary
    )
  end

  @doc "Metadata for attention weights emitted by Bumblebee outputs."
  def attention_weights(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.attn.hook_pattern",
      source_output: :attentions,
      capture_exactness: :bumblebee_output_attention_weights,
      capture_mode: :summary
    )
  end

  @doc "Builds canonical metadata for surface-declared nodes when known."
  def surface_metadata(signal_type, layer_index) do
    case {signal_type, layer_index} do
      {:final_logits, :final} ->
        final_logits()

      {:embeddings, nil} ->
        metadata("hook_embed", capture_exactness: :surface_declared)

      {:attention_q, layer} when is_integer(layer) ->
        metadata("blocks.#{layer}.attn.hook_q")

      {:attention_k, layer} when is_integer(layer) ->
        metadata("blocks.#{layer}.attn.hook_k")

      {:attention_v, layer} when is_integer(layer) ->
        metadata("blocks.#{layer}.attn.hook_v")

      {:attention_weights, layer} when is_integer(layer) ->
        attention_weights(layer)

      {:attention_maps, layer} when is_integer(layer) ->
        attention_weights(layer)

      {:head_outputs, layer} when is_integer(layer) ->
        metadata("blocks.#{layer}.hook_attn_out")

      {:middle_residuals, layer} when is_integer(layer) ->
        metadata("blocks.#{layer}.hook_mlp_out")

      {:late_residuals, :final} ->
        metadata("ln_final.hook_normalized")

      _other ->
        %{}
    end
  end

  defp metadata(activation_name, attrs \\ []) do
    ActivationMetadata.put_activation(%{}, activation_name, attrs)
  end
end
