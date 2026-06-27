defmodule CrucibleBumblebee.ActivationMapper do
  @moduledoc """
  Maps Bumblebee output keys and surface node names to canonical activation names.

  This module describes activations emitted by Bumblebee model outputs and by
  provider surface declarations. Deep attention, MLP, and residual activations
  are available when using the North-Shore-AI Bumblebee fork pinned by this
  package.
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

  def attention_query(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.attn.hook_q",
      source_output: :attention_queries,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def attention_key(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.attn.hook_k",
      source_output: :attention_keys,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def attention_value(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.attn.hook_v",
      source_output: :attention_values,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def attention_scores(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.attn.hook_attn_scores",
      source_output: :attention_scores,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def attention_z(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.attn.hook_z",
      source_output: :attention_zs,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def attention_output(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.hook_attn_out",
      source_output: :attention_outputs,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def mlp_pre(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.mlp.hook_pre",
      source_output: :mlp_pre_activations,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def mlp_post(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.mlp.hook_post",
      source_output: :mlp_post_activations,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def mlp_output(layer) when is_integer(layer) and layer >= 0 do
    metadata("blocks.#{layer}.hook_mlp_out",
      source_output: :mlp_outputs,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def residual_stream(layer, hook, source_output)
      when is_integer(layer) and layer >= 0 and
             hook in [:hook_resid_pre, :hook_resid_mid, :hook_resid_post] do
    metadata("blocks.#{layer}.#{hook}",
      source_output: source_output,
      capture_exactness: :bumblebee_deep_output,
      capture_mode: :summary
    )
  end

  def output_metadata(:attention_queries, layer), do: attention_query(layer)
  def output_metadata(:attention_keys, layer), do: attention_key(layer)
  def output_metadata(:attention_values, layer), do: attention_value(layer)
  def output_metadata(:attention_scores, layer), do: attention_scores(layer)
  def output_metadata(:attention_zs, layer), do: attention_z(layer)
  def output_metadata(:attention_outputs, layer), do: attention_output(layer)
  def output_metadata(:mlp_pre_activations, layer), do: mlp_pre(layer)
  def output_metadata(:mlp_post_activations, layer), do: mlp_post(layer)
  def output_metadata(:mlp_outputs, layer), do: mlp_output(layer)

  def output_metadata(:residual_streams_pre, layer),
    do: residual_stream(layer, :hook_resid_pre, :residual_streams_pre)

  def output_metadata(:residual_streams_mid, layer),
    do: residual_stream(layer, :hook_resid_mid, :residual_streams_mid)

  def output_metadata(:residual_streams_post, layer),
    do: residual_stream(layer, :hook_resid_post, :residual_streams_post)

  @doc "Builds canonical metadata for surface-declared nodes when known."
  def surface_metadata(signal_type, layer_index) do
    case {signal_type, layer_index} do
      {:final_logits, :final} ->
        final_logits()

      {:embeddings, nil} ->
        metadata("hook_embed", capture_exactness: :surface_declared)

      {:attention_q, layer} when is_integer(layer) ->
        attention_query(layer)

      {:attention_k, layer} when is_integer(layer) ->
        attention_key(layer)

      {:attention_v, layer} when is_integer(layer) ->
        attention_value(layer)

      {:attention_scores, layer} when is_integer(layer) ->
        attention_scores(layer)

      {:attention_weights, layer} when is_integer(layer) ->
        attention_weights(layer)

      {:attention_maps, layer} when is_integer(layer) ->
        attention_weights(layer)

      {:head_outputs, layer} when is_integer(layer) ->
        attention_z(layer)

      {:mlp_gates, layer} when is_integer(layer) ->
        mlp_pre(layer)

      {:middle_residuals, layer} when is_integer(layer) ->
        mlp_output(layer)

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
