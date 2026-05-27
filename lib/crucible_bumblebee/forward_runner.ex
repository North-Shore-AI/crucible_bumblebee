defmodule CrucibleBumblebee.ForwardRunner do
  @moduledoc """
  Runs a Bumblebee-style predict function and returns a Crucible forward trace.
  """

  alias CrucibleBumblebee.{CacheSummary, Qwen3Surface, SignalExtractor, TapCompiler}
  alias CrucibleSignalTrace.ForwardTrace

  def run(predict_fun, inputs, tap_plan, opts \\ []) when is_function(predict_fun, 1) do
    trace_id = Keyword.get(opts, :trace_id, "trace:#{System.unique_integer([:positive])}")
    model_ref = Keyword.get(opts, :model_ref, "bumblebee:model")
    surface = Keyword.get(opts, :surface, Qwen3Surface.surface(num_blocks: 1))

    with {:ok, compiled_taps} <- TapCompiler.compile(tap_plan, surface) do
      outputs = predict_fun.(inputs)

      {records, trajectory} =
        SignalExtractor.extract(outputs, trace_id: trace_id, model_ref: model_ref)

      {:ok,
       ForwardTrace.new!(
         trace_id: trace_id,
         model_ref: model_ref,
         tap_plan_ref: tap_plan.plan_id,
         signal_records: records,
         layer_trajectory: trajectory,
         final_logits: final_logits_ref(records),
         cache_summary: CacheSummary.summarize(Map.get(outputs, :cache)),
         metadata: %{compiled_taps: compiled_taps}
       )}
    end
  end

  defp final_logits_ref(records) do
    records
    |> Enum.find(&(&1.signal_ref.signal_type == :final_logits))
    |> case do
      nil -> nil
      record -> record.signal_ref
    end
  end
end
