defmodule CrucibleBumblebee.HookRegistry do
  @moduledoc """
  Axon graph and Crucible surface hook registry for internals probes.

  The registry reports what is actually present in the loaded Axon graph. It
  does not infer that a research-level signal is available unless a concrete
  graph node can be selected for a passive hook attempt.
  """

  alias CrucibleBumblebee.{ModelBundle, ModelSurface}
  alias CrucibleTap.{Surface, TapSelector}

  @type graph_node :: map()
  @type candidate :: map()

  @candidate_order [
    :embeddings,
    :final_logits,
    :hidden_state,
    :attention_weights,
    :residual_stream,
    :mlp_activation
  ]

  @spec list_nodes(struct()) :: [graph_node()]
  def list_nodes(%ModelBundle{model: model}), do: list_nodes(model)

  def list_nodes(%Axon{} = model) do
    {nodes, _counts} =
      Axon.reduce_nodes(model, {[], %{}}, fn node, {rows, counts} ->
        op_name = node.op_name || :unknown
        generated_name = generated_name(node.name, op_name, counts)
        counts = Map.update(counts, op_name, 1, &(&1 + 1))

        row = %{
          id: node.id,
          generated_name: generated_name,
          op_name: op_name,
          parents: normalize_parents(node.parent),
          parameter_names: parameter_names(node.parameters),
          option_keys: option_keys(node.opts),
          global_options: List.wrap(node.global_options),
          hook_count: length(List.wrap(node.hooks)),
          candidate_signals: candidate_signals(node)
        }

        {[row | rows], counts}
      end)

    Enum.reverse(nodes)
  end

  def list_nodes(%ModelSurface{surface: %Surface{} = surface}), do: list_nodes(surface)

  def list_nodes(%Surface{} = surface) do
    Enum.map(surface.nodes, fn node ->
      %{
        id: node.id,
        generated_name: node.layer_name || node.id,
        op_name: :surface_node,
        parents: [],
        parameter_names: [],
        option_keys: [],
        global_options: [],
        hook_count: 0,
        candidate_signals: [node.signal_type],
        surface_signal_type: node.signal_type,
        layer_index: node.layer_index,
        capture_modes: node.capture_modes,
        operations: node.operations
      }
    end)
  end

  @spec candidate_taps(ModelBundle.t() | struct() | [graph_node()]) :: [candidate()]
  def candidate_taps(%ModelBundle{} = bundle), do: bundle |> list_nodes() |> candidate_taps()
  def candidate_taps(%Axon{} = model), do: model |> list_nodes() |> candidate_taps()

  def candidate_taps(nodes) when is_list(nodes) do
    candidates =
      @candidate_order
      |> Enum.flat_map(fn signal ->
        nodes
        |> Enum.filter(&(signal in Map.get(&1, :candidate_signals, [])))
        |> preferred_candidate(signal)
        |> List.wrap()
      end)

    if Enum.any?(candidates, &(&1.target_signal == :final_logits)) do
      candidates
    else
      nodes
      |> Enum.filter(&(Map.get(&1, :op_name) == :dense))
      |> final_logits_fallback()
      |> case do
        nil -> candidates
        node -> candidates ++ [as_candidate(node, :final_logits)]
      end
    end
  end

  defp as_candidate(node, signal) do
    Map.merge(node, %{
      target_signal: signal,
      selector: %{node_id: node.id, generated_name: node.generated_name}
    })
  end

  defp preferred_candidate(nodes, _signal) when nodes in [[], nil], do: nil

  defp preferred_candidate(nodes, signal) do
    node =
      case signal do
        :final_logits ->
          Enum.find(nodes, &(Map.get(&1, :op_name) == :dense_transposed)) || List.last(nodes)

        :hidden_state ->
          Enum.find(nodes, &(:output_hidden_states in Map.get(&1, :global_options, []))) ||
            List.last(nodes)

        :attention_weights ->
          Enum.find(nodes, &(:output_attentions in Map.get(&1, :global_options, []))) ||
            List.last(nodes)

        :residual_stream ->
          List.last(nodes)

        :mlp_activation ->
          mlp_fallback(nodes)

        _other ->
          hd(nodes)
      end

    as_candidate(node, signal)
  end

  defp final_logits_fallback([]), do: nil

  defp final_logits_fallback(nodes) do
    Enum.find(nodes, &classification_head?(&1.generated_name)) || hd(nodes)
  end

  defp mlp_fallback([]), do: nil

  defp mlp_fallback(nodes) do
    Enum.find(nodes, &mlp_name?(&1.generated_name)) ||
      generic_mlp_fallback(nodes)
  end

  defp generic_mlp_fallback(nodes) do
    non_head = Enum.reject(nodes, &classification_head?(&1.generated_name))

    cond do
      length(non_head) > 1 -> List.last(non_head)
      non_head != [] -> hd(non_head)
      true -> hd(nodes)
    end
  end

  defp classification_head?(name) when is_binary(name) do
    String.contains?(name, [
      "classification_head.output",
      "classifier",
      "score",
      "logits",
      "head.output"
    ])
  end

  defp classification_head?(_name), do: false

  defp mlp_name?(name) when is_binary(name) do
    String.contains?(name, [
      ".ffn.output",
      ".ffn.intermediate",
      ".mlp.output",
      ".mlp.gate",
      "feed_forward"
    ])
  end

  defp mlp_name?(_name), do: false

  @spec resolve_tap(ModelSurface.t() | Surface.t(), TapSelector.t() | map() | keyword()) ::
          {:ok, list()} | {:error, term()}
  def resolve_tap(%ModelSurface{surface: %Surface{} = surface}, selector),
    do: resolve_tap(surface, selector)

  def resolve_tap(%Surface{} = surface, selector) do
    selector =
      case selector do
        %TapSelector{} = selector -> selector
        attrs -> TapSelector.new!(attrs)
      end

    case Surface.matching_nodes(surface, selector) do
      [] -> {:error, {:tap_not_found, selector}}
      nodes -> {:ok, nodes}
    end
  end

  defp candidate_signals(node) do
    op = node.op_name
    global_options = List.wrap(node.global_options)
    option_keys = option_keys(node.opts)

    []
    |> maybe_add(op == :embedding, :embeddings)
    |> maybe_add(op == :dense_transposed, :final_logits)
    |> maybe_add(op == :dense, :mlp_activation)
    |> maybe_add(op == :add, :residual_stream)
    |> maybe_add(op == :layer_norm, :hidden_state)
    |> maybe_add(:output_hidden_states in global_options, :hidden_state)
    |> maybe_add(:output_attentions in global_options, :attention_weights)
    |> maybe_add(op == :custom and attention_custom?(option_keys), :attention_weights)
  end

  defp attention_custom?(option_keys) do
    :causal in option_keys or :scale in option_keys or :window_size in option_keys
  end

  defp maybe_add(signals, true, signal), do: [signal | signals]
  defp maybe_add(signals, false, _signal), do: signals

  defp generated_name(name_fun, op_name, counts) when is_function(name_fun, 2) do
    name_fun.(op_name, counts)
  rescue
    _error -> "#{op_name}_unknown"
  end

  defp generated_name(name, _op_name, _counts) when is_binary(name), do: name
  defp generated_name(_name, op_name, counts), do: "#{op_name}_#{Map.get(counts, op_name, 0)}"

  defp normalize_parents(parents) when is_list(parents),
    do: Enum.map(parents, &normalize_parent/1)

  defp normalize_parents(parent) when is_integer(parent), do: [parent]
  defp normalize_parents(_other), do: []

  defp normalize_parent(parent) when is_integer(parent), do: parent
  defp normalize_parent(parent), do: inspect(parent, charlists: :as_lists)

  defp parameter_names(parameters) when is_map(parameters), do: Map.keys(parameters)

  defp parameter_names(parameters) when is_list(parameters) do
    Enum.map(parameters, fn
      %{name: name} -> name
      {key, _value} when is_atom(key) -> key
      other -> inspect(other)
    end)
  end

  defp parameter_names(_parameters), do: []

  defp option_keys(opts) when is_map(opts), do: Map.keys(opts)

  defp option_keys(opts) when is_list(opts) do
    Enum.map(opts, fn
      {key, _value} when is_atom(key) -> key
      other -> inspect(other)
    end)
  end

  defp option_keys(_opts), do: []
end
