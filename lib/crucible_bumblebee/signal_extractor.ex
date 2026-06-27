defmodule CrucibleBumblebee.SignalExtractor do
  @moduledoc """
  Extracts bounded Crucible signal records from Bumblebee-style model outputs.
  """

  alias Crucible.{SignalRecord, TensorSummary}
  alias CrucibleBumblebee.ActivationMapper
  alias CrucibleSignalTrace.LayerTrajectory

  def extract(outputs, attrs) when is_map(outputs) do
    trace_id = Keyword.fetch!(attrs, :trace_id)
    model_id = Keyword.fetch!(attrs, :model_id)

    records =
      []
      |> maybe_add_logits(outputs, trace_id, model_id)
      |> maybe_add_hidden_states(outputs, trace_id, model_id)
      |> maybe_add_attentions(outputs, trace_id, model_id)

    {Enum.reverse(records), trajectory(records)}
  end

  defp maybe_add_logits(records, %{logits: logits}, trace_id, model_id) do
    [
      record(trace_id, "final_logits", :final_logits, model_id, logits,
        layer_index: :final,
        node_name: "final_logits",
        metadata: ActivationMapper.final_logits()
      )
      | records
    ]
  end

  defp maybe_add_logits(records, _outputs, _trace_id, _model_id), do: records

  defp maybe_add_hidden_states(records, %{hidden_states: hidden_states}, trace_id, model_id) do
    hidden_states
    |> tuple_or_list()
    |> Enum.with_index()
    |> Enum.reduce(records, fn {hidden_state, index}, acc ->
      size = length(tuple_or_list(hidden_states))
      type = hidden_type(index, size)
      metadata = ActivationMapper.hidden_state(index, size)

      [
        record(trace_id, "hidden_states:#{index}", type, model_id, hidden_state,
          layer_index: Map.get(metadata, :layer_index, index),
          node_name: "hidden_states:#{index}",
          metadata: metadata
        )
        | acc
      ]
    end)
  end

  defp maybe_add_hidden_states(records, _outputs, _trace_id, _model_id), do: records

  defp maybe_add_attentions(records, %{attentions: attentions}, trace_id, model_id) do
    attentions
    |> tuple_or_list()
    |> Enum.with_index()
    |> Enum.reduce(records, fn {attention, index}, acc ->
      [
        record(trace_id, "attentions:#{index}", :attention_weights, model_id, attention,
          layer_index: index,
          node_name: "attentions:#{index}",
          metadata: ActivationMapper.attention_weights(index)
        )
        | acc
      ]
    end)
  end

  defp maybe_add_attentions(records, _outputs, _trace_id, _model_id), do: records

  defp record(trace_id, signal_id, signal_type, model_id, value, opts) do
    summary = TensorSummary.compute(value, entropy: signal_type == :final_logits)

    SignalRecord.new!(
      trace_id: trace_id,
      signal_id: signal_id,
      signal_type: signal_type,
      model_id: model_id,
      layer_index: Keyword.get(opts, :layer_index),
      node_name: Keyword.get(opts, :node_name),
      capture_method: :bumblebee_output,
      capability_status: :captured,
      dtype: summary.dtype,
      shape: summary.shape,
      rank: summary.rank,
      tensor_summary: summary,
      metadata: Keyword.get(opts, :metadata, %{})
    )
  end

  defp trajectory(records) do
    points =
      records
      |> Enum.filter(
        &(&1.signal_type in [
            :embeddings,
            :early_residuals,
            :middle_residuals,
            :late_residuals
          ])
      )
      |> Enum.map(fn record ->
        %{
          layer_index: record.layer_index,
          signal_ref: record.signal_id,
          norm: record.tensor_summary.norm_l2
        }
      end)

    LayerTrajectory.new!(points)
  end

  defp tuple_or_list(value) when is_tuple(value), do: Tuple.to_list(value)
  defp tuple_or_list(value) when is_list(value), do: value

  defp hidden_type(0, _size), do: :embeddings
  defp hidden_type(index, size) when index == size - 1, do: :late_residuals
  defp hidden_type(index, _size) when index <= 2, do: :early_residuals
  defp hidden_type(_index, _size), do: :middle_residuals
end
