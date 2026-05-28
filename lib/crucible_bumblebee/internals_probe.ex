defmodule CrucibleBumblebee.InternalsProbe do
  @moduledoc """
  Native internals probe.

  The probe dumps Axon graph metadata, selects concrete hook candidates, and
  attempts a passive hooked forward pass. Captured values are summarized only;
  raw tensors are not written to JSON artifacts.
  """

  alias CrucibleBumblebee.{
    Artifacts,
    HookRegistry,
    ModelBundle,
    ModelLoader,
    ModelLoader.Options
  }

  @prompt "Hi"
  @selected_signals [
    :embeddings,
    :hidden_state,
    :attention_weights,
    :residual_stream,
    :mlp_activation,
    :final_logits
  ]
  @derived_signals [
    :intermediate_logits,
    :logit_lens_projection,
    :activation_cache,
    :router_logits,
    :moe_expert_weights,
    :kv_cache_metadata,
    :active_residual_injection
  ]

  @spec run_model(map(), keyword()) :: map()
  def run_model(model, opts \\ []) when is_map(model) do
    root = Keyword.get(opts, :artifact_root)
    backend = Keyword.get(opts, :backend, :binary)
    prompt = Keyword.get(opts, :prompt, @prompt)
    started = System.monotonic_time(:millisecond)

    Artifacts.ensure_layout!(root: root)

    try do
      bundle =
        ModelLoader.load!(
          Options.new(
            model_id: model.model_id,
            tokenizer_id: Map.get(model, :tokenizer_id, model.model_id),
            backend: backend,
            architecture: model.architecture,
            module: Map.get(model, :module),
            artifact_root: root
          )
        )

      tokenizer = Bumblebee.configure(bundle.tokenizer, return_token_type_ids: false)
      inputs = Bumblebee.apply_tokenizer(tokenizer, prompt)

      run_bundle(bundle, inputs, model, Keyword.merge(opts, backend: backend, started: started))
    rescue
      error ->
        row =
          base_row(model, backend, :graph_dump, "blocked_expected",
            reason: Exception.message(error),
            duration_ms: elapsed_ms(started)
          )

        write_row!(row, root)
        %{ok: false, rows: [row], graph_path: nil, activation_cache_path: nil}
    end
  end

  @spec run_bundle(ModelBundle.t(), map() | Nx.Tensor.t(), map(), keyword()) :: map()
  def run_bundle(%ModelBundle{} = bundle, inputs, model, opts \\ []) when is_map(model) do
    root = Keyword.get(opts, :artifact_root)
    backend = Keyword.get(opts, :backend, bundle.backend)
    started = Keyword.get(opts, :started, System.monotonic_time(:millisecond))

    Artifacts.ensure_layout!(root: root)

    graph_nodes = HookRegistry.list_nodes(bundle)
    candidates = HookRegistry.candidate_taps(graph_nodes)
    graph_path = write_graph!(model, backend, bundle, graph_nodes, candidates, root)

    {captured_rows, captured_summaries} =
      capture_candidates(bundle, inputs, model, backend, candidates, graph_path)

    derived_rows =
      derived_rows(model, backend, captured_rows, captured_summaries, root, graph_path)

    rows =
      [
        base_row(model, backend, :graph_dump, "captured",
          reason: nil,
          graph_path: graph_path,
          node_count: length(graph_nodes),
          candidate_count: length(candidates),
          duration_ms: elapsed_ms(started)
        )
        | captured_rows ++ derived_rows
      ]

    Enum.each(rows, &write_row!(&1, root))

    activation_cache_path =
      maybe_write_activation_cache!(model, backend, captured_summaries, root)

    %{
      ok: Enum.any?(captured_rows, &(&1.status == "captured")),
      rows: rows,
      graph_path: graph_path,
      activation_cache_path: activation_cache_path
    }
  end

  defp capture_candidates(bundle, inputs, model, backend, candidates, graph_path) do
    selected = select_candidates(candidates)
    ref = make_ref()
    pid = self()
    node_ids = MapSet.new(Enum.map(selected, & &1.id))
    candidate_by_id = Map.new(selected, &{&1.id, &1})

    hooked_model =
      Axon.map_nodes(bundle.model, fn node ->
        if MapSet.member?(node_ids, node.id) do
          candidate = Map.fetch!(candidate_by_id, node.id)
          attach_capture_hook(node, pid, ref, candidate)
        else
          node
        end
      end)

    capture_result =
      try do
        _outputs = Axon.predict(hooked_model, bundle.params, inputs)
        {:ok, collect_hook_messages(ref, map_size(candidate_by_id), [])}
      rescue
        error -> {:error, Exception.message(error)}
      end

    case capture_result do
      {:ok, messages} ->
        summaries_by_signal = Map.new(messages, &{&1.target_signal, &1.summary})

        rows =
          Enum.map(selected, fn candidate ->
            summary = Map.get(summaries_by_signal, candidate.target_signal)
            captured_row(model, backend, candidate, summary, graph_path)
          end)

        missing_rows = missing_unselected_rows(model, backend, selected, graph_path)

        rows = rows ++ missing_rows

        {rows, summaries_by_signal}

      {:error, reason} ->
        rows =
          Enum.map(selected, fn candidate ->
            base_row(model, backend, candidate.target_signal, "failed_with_exception",
              reason: reason,
              graph_path: graph_path,
              node_id: candidate.id,
              generated_name: candidate.generated_name,
              op_name: candidate.op_name
            )
          end)

        {rows, %{}}
    end
  end

  defp select_candidates(candidates) do
    candidates
    |> Enum.filter(&(&1.target_signal in @selected_signals))
    |> Enum.uniq_by(& &1.target_signal)
  end

  defp attach_capture_hook(node, pid, ref, candidate) do
    Axon.attach_hook(node, fn value ->
      send(pid, {
        :crucible_bumblebee_internals_hook,
        ref,
        %{
          node_id: candidate.id,
          target_signal: candidate.target_signal,
          generated_name: candidate.generated_name,
          op_name: candidate.op_name,
          summary: summarize_value(value)
        }
      })

      value
    end)
  end

  defp collect_hook_messages(_ref, expected, acc) when expected <= length(acc),
    do: Enum.reverse(acc)

  defp collect_hook_messages(ref, expected, acc) do
    receive do
      {:crucible_bumblebee_internals_hook, ^ref, payload} ->
        collect_hook_messages(ref, expected, [payload | acc])
    after
      250 ->
        Enum.reverse(acc)
    end
  end

  defp captured_row(model, backend, candidate, nil, graph_path) do
    base_row(model, backend, candidate.target_signal, "failed_with_exception",
      reason: :hook_message_timeout,
      graph_path: graph_path,
      node_id: candidate.id,
      generated_name: candidate.generated_name,
      op_name: candidate.op_name
    )
  end

  defp captured_row(model, backend, candidate, %{kind: "axon_none"} = summary, graph_path) do
    base_row(model, backend, candidate.target_signal, "blocked_by_bumblebee_api",
      reason: :axon_none,
      graph_path: graph_path,
      node_id: candidate.id,
      generated_name: candidate.generated_name,
      op_name: candidate.op_name,
      value_summary: summary
    )
  end

  defp captured_row(model, backend, candidate, summary, graph_path) do
    status =
      if contains_tensor_summary?(summary) do
        "captured"
      else
        "blocked_by_axon_graph"
      end

    reason = if(status == "captured", do: nil, else: :no_tensor_value)

    base_row(model, backend, candidate.target_signal, status,
      reason: reason,
      graph_path: graph_path,
      node_id: candidate.id,
      generated_name: candidate.generated_name,
      op_name: candidate.op_name,
      value_summary: summary
    )
  end

  defp missing_unselected_rows(model, backend, selected, graph_path) do
    selected_signals = MapSet.new(Enum.map(selected, & &1.target_signal))

    @selected_signals
    |> Enum.reject(&MapSet.member?(selected_signals, &1))
    |> Enum.map(fn signal ->
      base_row(model, backend, signal, "blocked_by_axon_graph",
        reason: :candidate_not_found,
        graph_path: graph_path
      )
    end)
  end

  defp derived_rows(model, backend, captured_rows, captured_summaries, root, graph_path) do
    captured_signals =
      captured_rows
      |> Enum.filter(&(&1.status == "captured"))
      |> Enum.map(& &1.signal)
      |> MapSet.new()

    Enum.map(@derived_signals, fn
      :intermediate_logits = signal ->
        derived_row(model, backend, signal, graph_path, captured_signals,
          required: [:hidden_state, :final_logits],
          blocked_reason: :hidden_states_unavailable
        )

      :logit_lens_projection = signal ->
        derived_row(model, backend, signal, graph_path, captured_signals,
          required: [:hidden_state, :final_logits],
          blocked_reason: :hidden_states_or_lm_head_projection_unavailable
        )

      :activation_cache = signal ->
        status =
          if activation_summaries?(captured_summaries),
            do: "captured",
            else: "blocked_by_axon_graph"

        reason = if(status == "captured", do: nil, else: :no_activation_signal_captured)

        base_row(model, backend, signal, status,
          reason: reason,
          graph_path: graph_path,
          artifact_path: activation_cache_path(model, backend, root)
        )

      :router_logits = signal ->
        base_row(model, backend, signal, "unsupported_by_model_family",
          reason: :no_moe_router_in_attempted_model,
          graph_path: graph_path
        )

      :moe_expert_weights = signal ->
        base_row(model, backend, signal, "unsupported_by_model_family",
          reason: :no_moe_router_in_attempted_model,
          graph_path: graph_path
        )

      :kv_cache_metadata = signal ->
        base_row(model, backend, signal, "blocked_by_bumblebee_api",
          reason: :cache_hidden_in_output_surface,
          graph_path: graph_path
        )

      :active_residual_injection = signal ->
        base_row(model, backend, signal, "unsupported_by_surface",
          reason: :provider_does_not_advertise_active_mutation,
          graph_path: graph_path
        )
    end)
  end

  defp derived_row(model, backend, signal, graph_path, captured_signals, opts) do
    required = Keyword.fetch!(opts, :required)

    if Enum.all?(required, &MapSet.member?(captured_signals, &1)) do
      base_row(model, backend, signal, "blocked_by_bumblebee_api",
        reason: :projection_not_exposed,
        graph_path: graph_path
      )
    else
      base_row(model, backend, signal, "blocked_by_bumblebee_api",
        reason: Keyword.fetch!(opts, :blocked_reason),
        graph_path: graph_path
      )
    end
  end

  defp write_graph!(model, backend, bundle, graph_nodes, candidates, root) do
    filename = "#{model.rung}_#{Artifacts.safe_name(model.model_id)}_#{backend}.graph.json"

    Artifacts.write_json!(
      :graphs,
      filename,
      %{
        schema: "crucible_bumblebee.v5.graph",
        model_id: bundle.model_id,
        model_family: bundle.model_family,
        backend: backend,
        spec_module: spec_module(bundle.spec),
        node_count: length(graph_nodes),
        op_counts: Axon.get_op_counts(bundle.model),
        candidate_count: length(candidates),
        candidates: candidates,
        nodes: graph_nodes
      },
      root: root
    )
  end

  defp maybe_write_activation_cache!(_model, _backend, captured_summaries, _root)
       when map_size(captured_summaries) == 0,
       do: nil

  defp maybe_write_activation_cache!(model, backend, captured_summaries, root) do
    summaries =
      captured_summaries
      |> Enum.filter(fn {signal, summary} ->
        signal in [
          :embeddings,
          :hidden_state,
          :attention_weights,
          :residual_stream,
          :mlp_activation
        ] and
          contains_tensor_summary?(summary)
      end)
      |> Map.new()

    if map_size(summaries) > 0 do
      path = activation_cache_path(model, backend, root)

      Artifacts.write_json!(
        :graphs,
        Path.basename(path),
        %{
          schema: "crucible_bumblebee.v5.activation_cache_summary",
          model_id: model.model_id,
          family: model.family,
          backend: backend,
          summaries: summaries
        },
        root: root
      )
    end
  end

  defp activation_cache_path(model, backend, root) do
    Artifacts.path!(
      :graphs,
      "#{model.rung}_#{Artifacts.safe_name(model.model_id)}_#{backend}.activation_cache.json",
      root: root
    )
  end

  defp write_row!(row, root) do
    Artifacts.append_jsonl!(:internals_matrix, "internals_ladder.jsonl", row, root: root)
  end

  defp base_row(model, backend, signal, status, opts) do
    %{
      rung: Map.get(model, :rung),
      model_id: Map.get(model, :model_id),
      family: Map.get(model, :family),
      backend: backend,
      signal: signal,
      status: status,
      result: if(status == "captured", do: "passed", else: "blocked_expected"),
      reason: stringify(Keyword.get(opts, :reason)),
      graph_path: Keyword.get(opts, :graph_path),
      artifact_path: Keyword.get(opts, :artifact_path),
      node_id: Keyword.get(opts, :node_id),
      generated_name: Keyword.get(opts, :generated_name),
      op_name: Keyword.get(opts, :op_name),
      node_count: Keyword.get(opts, :node_count),
      candidate_count: Keyword.get(opts, :candidate_count),
      duration_ms: Keyword.get(opts, :duration_ms),
      value_summary: Keyword.get(opts, :value_summary)
    }
  end

  defp summarize_value(%Nx.Tensor{} = tensor) do
    shape = tensor |> Nx.shape() |> Tuple.to_list()

    %{
      kind: "tensor",
      shape: shape,
      rank: length(shape),
      dtype: inspect(Nx.type(tensor))
    }
  end

  defp summarize_value(%Axon.None{}), do: %{kind: "axon_none"}

  defp summarize_value(value) when is_map(value) do
    entries =
      value
      |> Enum.map(fn {key, nested} -> {key, summarize_value(nested)} end)
      |> Map.new()

    %{kind: "map", keys: value |> Map.keys() |> Enum.map(&to_string/1), entries: entries}
  end

  defp summarize_value(value) when is_tuple(value) do
    entries = value |> Tuple.to_list() |> Enum.map(&summarize_value/1)
    %{kind: "tuple", size: tuple_size(value), entries: entries}
  end

  defp summarize_value(value) when is_list(value) do
    entries = Enum.map(value, &summarize_value/1)
    %{kind: "list", size: length(value), entries: entries}
  end

  defp summarize_value(value), do: %{kind: inspect(value)}

  defp contains_tensor_summary?(%{kind: "tensor"}), do: true

  defp contains_tensor_summary?(%{entries: entries}) when is_list(entries),
    do: Enum.any?(entries, &contains_tensor_summary?/1)

  defp contains_tensor_summary?(%{entries: entries}) when is_map(entries),
    do: entries |> Map.values() |> Enum.any?(&contains_tensor_summary?/1)

  defp contains_tensor_summary?(_summary), do: false

  defp activation_summaries?(summaries) do
    Enum.any?(summaries, fn {signal, summary} ->
      signal in [
        :embeddings,
        :hidden_state,
        :attention_weights,
        :residual_stream,
        :mlp_activation
      ] and
        contains_tensor_summary?(summary)
    end)
  end

  defp stringify(nil), do: nil
  defp stringify(value) when is_binary(value), do: value
  defp stringify(value), do: inspect(value)

  defp spec_module(%{__struct__: module}), do: inspect(module)
  defp spec_module(nil), do: nil
  defp spec_module(spec), do: inspect(spec)

  defp elapsed_ms(start_ms), do: System.monotonic_time(:millisecond) - start_ms
end
