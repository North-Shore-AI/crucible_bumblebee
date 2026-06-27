defmodule CrucibleBumblebee.ExampleSurface do
  @moduledoc """
  Generic fixture surface used by tests, examples, and documentation.
  """

  @behaviour CrucibleBumblebee.ModelSurface

  alias CrucibleBumblebee.ModelSurface

  def surface(opts \\ []), do: ModelSurface.from_module!(__MODULE__, opts)

  @impl true
  def id, do: :example_transformer

  @impl true
  def model_family, do: :example_transformer

  @impl true
  def capabilities(_opts \\ []) do
    %{
      hidden_states: true,
      attentions: true,
      deep_attention_activations: true,
      deep_mlp_activations: true,
      residual_streams: true,
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
  def output_options(_compiled_plan), do: [output_hidden_states: true, output_attentions: true]

  @impl true
  def preflight(_model_info, opts) do
    num_blocks = Keyword.get(opts, :num_blocks, 1)

    {:ok,
     %{
       surface_id: id(),
       model_family: model_family(),
       nodes: Enum.map(nodes(num_blocks), & &1[:id]),
       post_processing_extractors: [
         :final_logits,
         :hidden_states,
         :attentions,
         :deep_attention_activations,
         :deep_mlp_activations,
         :residual_streams,
         :cache
       ],
       logit_lens: %{
         final_norm: [:params, :decoder, :final_norm],
         unembedding: [:params, :lm_head, :kernel],
         hidden_states: :outputs_hidden_states
       },
       unsupported: []
     }}
  end

  @impl true
  def logit_lens_access(_model_info, params) when is_map(params) do
    with {:ok, final_norm} <- fetch_path(params, [:decoder, :final_norm]),
         {:ok, unembedding} <- fetch_path(params, [:lm_head, :kernel]) do
      {:ok,
       %{final_norm: final_norm, unembedding: unembedding, hidden_states: :outputs_hidden_states}}
    else
      {:error, _reason} -> {:error, :unsupported}
    end
  end

  def nodes(num_blocks) when is_integer(num_blocks) and num_blocks > 0 do
    [
      node("tokens.embedding", :embeddings, nil)
      | Enum.flat_map(0..(num_blocks - 1), &block_nodes/1)
    ] ++
      [
        node("decoder.final_norm", :late_residuals, :final),
        node("lm_head.output", :final_logits, :final)
      ]
  end

  defp block_nodes(layer) do
    [
      node("decoder.layers.#{layer}.attention.query", :attention_q, layer),
      node("decoder.layers.#{layer}.attention.key", :attention_k, layer),
      node("decoder.layers.#{layer}.attention.value", :attention_v, layer),
      node("decoder.layers.#{layer}.attention.weights", :attention_weights, layer),
      node("decoder.layers.#{layer}.attention.z", :head_outputs, layer),
      node(
        "decoder.layers.#{layer}.attention.output",
        :residual_stream,
        layer,
        CrucibleBumblebee.ActivationMapper.attention_output(layer)
      ),
      node("decoder.layers.#{layer}.pre_attention_norm", :norm_telemetry, layer),
      node("decoder.layers.#{layer}.mlp.gate", :mlp_gates, layer),
      node(
        "decoder.layers.#{layer}.mlp.pre",
        :mlp_activation,
        layer,
        CrucibleBumblebee.ActivationMapper.mlp_pre(layer)
      ),
      node(
        "decoder.layers.#{layer}.mlp.post",
        :mlp_activation,
        layer,
        CrucibleBumblebee.ActivationMapper.mlp_post(layer)
      ),
      node(
        "outputs.mlp_outputs.#{layer}",
        :residual_stream,
        layer,
        CrucibleBumblebee.ActivationMapper.mlp_output(layer)
      ),
      node("decoder.layers.#{layer}.mlp.output", :middle_residuals, layer, %{}),
      node(
        "decoder.layers.#{layer}.resid.pre",
        :residual_stream,
        layer,
        CrucibleBumblebee.ActivationMapper.residual_stream(
          layer,
          :hook_resid_pre,
          :residual_streams_pre
        )
      ),
      node(
        "decoder.layers.#{layer}.resid.mid",
        :residual_stream,
        layer,
        CrucibleBumblebee.ActivationMapper.residual_stream(
          layer,
          :hook_resid_mid,
          :residual_streams_mid
        )
      ),
      node(
        "decoder.layers.#{layer}.resid.post",
        :residual_stream,
        layer,
        CrucibleBumblebee.ActivationMapper.residual_stream(
          layer,
          :hook_resid_post,
          :residual_streams_post
        )
      )
    ]
  end

  defp node(layer_name, signal_type, layer_index, metadata \\ nil) do
    metadata =
      metadata || CrucibleBumblebee.ActivationMapper.surface_metadata(signal_type, layer_index)

    [
      id: layer_name,
      signal_type: signal_type,
      activation_name: Map.get(metadata, :activation_name),
      axes: Map.get(metadata, :axes),
      layer_name: layer_name,
      layer_index: layer_index,
      operations: operations(signal_type),
      capture_modes: [:summary, :sample, :compressed_vector],
      metadata: metadata
    ]
  end

  defp operations(signal_type)
       when signal_type in [
              :embeddings,
              :middle_residuals,
              :late_residuals,
              :attention_q,
              :attention_k,
              :attention_v,
              :attention_weights,
              :head_outputs,
              :residual_stream,
              :mlp_gates,
              :mlp_activation,
              :final_logits
            ],
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
