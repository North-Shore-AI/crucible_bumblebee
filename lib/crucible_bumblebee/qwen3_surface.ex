defmodule CrucibleBumblebee.Qwen3Surface do
  @moduledoc """
  Example Qwen3-family layer-name surface map for Bumblebee.

  This module is optional. Reusable runners consume the `ModelSurface`
  behaviour and do not assume Qwen-family params or layer names.
  """

  @behaviour CrucibleBumblebee.ModelSurface

  alias CrucibleBumblebee.ModelSurface

  @doc "Builds the example Qwen3-family surface."
  def surface(opts \\ []) do
    num_blocks = Keyword.get(opts, :num_blocks, 28)
    ModelSurface.from_module!(__MODULE__, Keyword.put(opts, :num_blocks, num_blocks))
  end

  @impl true
  def id, do: :qwen3_example

  @impl true
  def model_family, do: :qwen3

  @impl true
  def capabilities(_opts \\ []) do
    %{
      hidden_states: true,
      attentions: false,
      named_hooks: true,
      final_logits: true,
      cache_metadata: true,
      token_boundary_steering: true,
      logits_processors: true,
      logit_lens: true,
      in_graph_steering: false
    }
  end

  @impl true
  def output_options(_compiled_plan), do: [output_hidden_states: true]

  @impl true
  def preflight(_model_info, opts) do
    num_blocks = Keyword.get(opts, :num_blocks, 28)

    {:ok,
     %{
       surface_id: id(),
       model_family: model_family(),
       nodes: Enum.map(nodes(num_blocks), & &1[:id]),
       post_processing_extractors: [:final_logits, :hidden_states, :cache],
       logit_lens: %{
         final_norm: [:params, :output_norm],
         unembedding: [:params, :language_modeling_head, :output, :kernel],
         hidden_states: :outputs_hidden_states
       },
       unsupported: [:in_graph_steering]
     }}
  end

  @impl true
  def logit_lens_access(_model_info, params) when is_map(params) do
    with {:ok, final_norm} <- fetch_path(params, [:output_norm]),
         {:ok, unembedding} <- fetch_path(params, [:language_modeling_head, :output, :kernel]) do
      {:ok,
       %{final_norm: final_norm, unembedding: unembedding, hidden_states: :outputs_hidden_states}}
    else
      {:error, _reason} -> {:error, :unsupported}
    end
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
    metadata = CrucibleBumblebee.ActivationMapper.surface_metadata(signal_type, layer_index)

    [
      id: layer_name,
      signal_type: signal_type,
      activation_name: Map.get(metadata, :activation_name),
      axes: Map.get(metadata, :axes),
      layer_name: layer_name,
      layer_index: layer_index,
      operations: operations(signal_type),
      capture_modes: [:summary, :sample],
      metadata: metadata
    ]
  end

  defp operations(signal_type)
       when signal_type in [:embeddings, :middle_residuals, :late_residuals, :final_logits],
       do: [:read, :probe]

  defp operations(_signal_type), do: [:probe]

  defp fetch_path(value, []), do: {:ok, value}

  defp fetch_path(value, [key | rest]) when is_map(value) do
    cond do
      Map.has_key?(value, key) -> fetch_path(Map.fetch!(value, key), rest)
      Map.has_key?(value, to_string(key)) -> fetch_path(Map.fetch!(value, to_string(key)), rest)
      true -> {:error, {:missing_path, key}}
    end
  end

  defp fetch_path(_value, [key | _rest]), do: {:error, {:missing_path, key}}
end
