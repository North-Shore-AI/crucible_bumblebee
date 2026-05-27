defmodule CrucibleBumblebee.Qwen3Surface do
  @moduledoc """
  Qwen3 layer-name surface map for the local Bumblebee implementation.
  """

  alias CrucibleBumblebee.ModelSurface

  @doc "Builds the Qwen3 surface. Defaults to 28 blocks for Qwen3 0.6B."
  def surface(opts \\ []) do
    num_blocks = Keyword.get(opts, :num_blocks, 28)
    ModelSurface.new!(:qwen3, nodes(num_blocks), %{num_blocks: num_blocks})
  end

  def nodes(num_blocks) when is_integer(num_blocks) and num_blocks > 0 do
    [
      node("embedder.token_embedding", :embeddings, nil)
      | Enum.flat_map(0..(num_blocks - 1), &block_nodes/1)
    ] ++
      [
        node("output_norm", :late_residuals, :final),
        node("language_modeling_head.output", :final_logits, :final)
      ]
  end

  defp block_nodes(layer) do
    [
      node("decoder.blocks.#{layer}.self_attention.query", :attention_q, layer),
      node("decoder.blocks.#{layer}.self_attention.key", :attention_k, layer),
      node("decoder.blocks.#{layer}.self_attention.value", :attention_v, layer),
      node("decoder.blocks.#{layer}.self_attention.output", :head_outputs, layer),
      node("decoder.blocks.#{layer}.self_attention_norm", :norm_telemetry, layer),
      node("decoder.blocks.#{layer}.ffn.gate", :mlp_gates, layer),
      node("decoder.blocks.#{layer}.ffn.intermediate", :middle_residuals, layer),
      node("decoder.blocks.#{layer}.ffn.output", :middle_residuals, layer),
      node("decoder.blocks.#{layer}.output_norm", :norm_telemetry, layer)
    ]
  end

  defp node(layer_name, signal_type, layer_index) do
    [
      id: layer_name,
      signal_type: signal_type,
      layer_name: layer_name,
      layer_index: layer_index,
      operations: [:read, :probe],
      capture_modes: [:summary, :sample]
    ]
  end
end
