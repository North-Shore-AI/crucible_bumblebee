defmodule CrucibleBumblebee.LogitLensRunner do
  @moduledoc """
  Provider-side logit-lens projection from Crucible traces and Bumblebee params.

  The runner relies on `ModelSurface.logit_lens_access/3` for model-family
  parameter paths. It never hard-codes Qwen, GPT-2, or fixture parameter shapes.
  """

  alias Crucible.{SignalRecord, TensorSummary}
  alias CrucibleBumblebee.ModelSurface
  alias CrucibleMechInterp.{ActivationCache, LogitLens}

  @doc "Projects trace residual activations through the model unembedding."
  def project_trace(
        %Crucible.ForwardTrace{} = trace,
        %ModelSurface{} = surface,
        model_info,
        params,
        opts \\ []
      ) do
    with {:ok, lens_info} <- ModelSurface.logit_lens_access(surface, model_info, params),
         {:ok, cache} <- activation_cache(trace, lens_info, model_info, opts),
         :ok <- ensure_tensor_cache(cache) do
      {:ok, LogitLens.project(cache, opts)}
    end
  end

  @doc "Projects raw Bumblebee output hidden states through the model unembedding."
  def project_outputs(outputs, %ModelSurface{} = surface, model_info, params, opts \\ [])
      when is_map(outputs) do
    with {:ok, lens_info} <- ModelSurface.logit_lens_access(surface, model_info, params),
         {:ok, cache} <- output_cache(outputs, lens_info, model_info, opts) do
      {:ok, LogitLens.project(cache, opts)}
    end
  end

  @doc "Emits bounded `:logit_lens_intermediate` signal records for a trace."
  def intermediate_records(
        %Crucible.ForwardTrace{} = trace,
        %ModelSurface{} = surface,
        model_info,
        params,
        opts \\ []
      ) do
    with {:ok, {logits, labels}} <- project_trace(trace, surface, model_info, params, opts) do
      {:ok,
       labels
       |> Enum.with_index()
       |> Enum.map(fn {label, index} ->
         logits
         |> Nx.slice_along_axis(index, 1, axis: 0)
         |> signal_record(trace, label, index, opts)
       end)}
    end
  end

  @doc "Projects raw outputs and emits bounded `:logit_lens_intermediate` records."
  def intermediate_records_from_outputs(
        outputs,
        trace_attrs,
        %ModelSurface{} = surface,
        model_info,
        params,
        opts \\ []
      ) do
    with {:ok, {logits, labels}} <- project_outputs(outputs, surface, model_info, params, opts) do
      trace = struct(Crucible.ForwardTrace, Map.new(trace_attrs))

      {:ok,
       labels
       |> Enum.with_index()
       |> Enum.map(fn {label, index} ->
         logits
         |> Nx.slice_along_axis(index, 1, axis: 0)
         |> signal_record(trace, label, index, opts)
       end)}
    end
  end

  defp activation_cache(trace, lens_info, model_info, opts) do
    n_layers =
      Keyword.get(opts, :n_layers) ||
        Map.get(model_info, :n_layers) ||
        Map.get(model_info, "n_layers") ||
        infer_n_layers(trace)

    cache_model_info =
      lens_info
      |> Map.take([:final_norm, :unembedding, :unembedding_bias, :unembedding_orientation])
      |> Map.put(:n_layers, n_layers)

    ActivationCache.from_trace(trace, model_info: cache_model_info)
  end

  defp ensure_tensor_cache(%ActivationCache{} = cache) do
    cache.activations
    |> Enum.find(fn {name, value} ->
      String.contains?(name, "hook_resid") and not match?(%Nx.Tensor{}, value)
    end)
    |> case do
      nil -> :ok
      {name, _value} -> {:error, {:raw_activations_required, name}}
    end
  end

  defp infer_n_layers(%Crucible.ForwardTrace{} = trace) do
    trace.signals
    |> Enum.map(fn %SignalRecord{} = signal ->
      signal.layer_index || Map.get(signal.metadata, :layer_index)
    end)
    |> Enum.filter(&is_integer/1)
    |> case do
      [] -> raise ArgumentError, "cannot infer n_layers from trace; pass :n_layers in opts"
      indices -> Enum.max(indices) + 1
    end
  end

  defp output_cache(outputs, lens_info, model_info, opts) do
    hidden_states = Map.get(outputs, :hidden_states) || Map.get(outputs, "hidden_states")

    if hidden_states in [nil, []] do
      {:error, :hidden_states_unavailable}
    else
      hidden_states = tuple_or_list(hidden_states)

      n_layers =
        Keyword.get(opts, :n_layers) || Map.get(model_info, :n_layers) ||
          length(hidden_states) - 1

      activations =
        hidden_states
        |> Enum.with_index()
        |> Map.new(fn {tensor, index} ->
          metadata = CrucibleBumblebee.ActivationMapper.hidden_state(index, length(hidden_states))
          {metadata.activation_name, tensor}
        end)

      cache_model_info =
        lens_info
        |> Map.take([:final_norm, :unembedding, :unembedding_bias, :unembedding_orientation])
        |> Map.put(:n_layers, n_layers)

      {:ok, ActivationCache.new!(activations, model_info: cache_model_info)}
    end
  end

  defp tuple_or_list(value) when is_tuple(value), do: Tuple.to_list(value)
  defp tuple_or_list(value) when is_list(value), do: value

  defp signal_record(logits, trace, label, index, opts) do
    squeezed = Nx.squeeze(logits, axes: [0])
    summary = TensorSummary.compute(squeezed, entropy: false, top_k: Keyword.get(opts, :top_k, 5))

    SignalRecord.new!(
      trace_id: trace.trace_id,
      signal_id: "logit_lens:#{label}",
      signal_type: :logit_lens_intermediate,
      model_id: trace.model_id,
      model_family: trace.model_family,
      provider_kind: :elixir_bumblebee,
      backend: trace.backend,
      layer_index: index,
      node_name: "logit_lens:#{label}",
      capture_method: :logit_lens_projection,
      capability_status: :captured,
      dtype: summary.dtype,
      shape: summary.shape,
      rank: summary.rank,
      tensor_summary: summary,
      metadata: %{
        logit_lens_label: label,
        source_trace_id: trace.trace_id,
        source_provider: :crucible_bumblebee
      }
    )
  end
end
