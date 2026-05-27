defmodule CrucibleBumblebee.ForwardRunner do
  @moduledoc """
  Runs a Bumblebee-style predict function and returns a Crucible forward trace.
  """

  alias CrucibleBumblebee.{CacheSummary, ExampleSurface, Serving, SignalExtractor, TapCompiler}
  alias CrucibleSignalTrace.ForwardTrace

  def run(predict_fun, inputs, tap_plan, opts \\ []) when is_function(predict_fun, 1) do
    with {:ok, serving} <- compile_serving(predict_fun, tap_plan, opts) do
      run_serving(serving, inputs, opts)
    end
  end

  def run_serving(%Serving{} = serving, inputs, opts \\ []) do
    trace_id = Keyword.get(opts, :trace_id, "trace:#{System.unique_integer([:positive])}")
    model_ref = Keyword.get(opts, :model_ref, serving.model_ref || "bumblebee:model")
    outputs = serving.predict_fun.(inputs)

    {records, trajectory} =
      SignalExtractor.extract(outputs, trace_id: trace_id, model_ref: model_ref)

    {:ok,
     ForwardTrace.new!(
       trace_id: trace_id,
       model_ref: model_ref,
       tap_plan_ref: serving.compiled_taps.tap_plan_id,
       signal_records: records,
       layer_trajectory: trajectory,
       final_logits: final_logits_ref(records),
       cache_summary: CacheSummary.summarize(Map.get(outputs, :cache)),
       metadata: %{
         compiled_taps: compiled_tap_summary(serving.compiled_taps),
         serving_ref: serving.serving_ref,
         lifecycle: [:plan_compilation, :serving_compilation, :execution]
       }
     )}
  end

  def compile_serving(predict_fun, tap_plan, opts \\ []) when is_function(predict_fun, 1) do
    model_ref = Keyword.get(opts, :model_ref, "bumblebee:model")
    surface = Keyword.get(opts, :surface, ExampleSurface.surface(num_blocks: 1))

    with {:ok, compiled_taps} <- TapCompiler.compile(tap_plan, surface) do
      {:ok,
       %Serving{
         serving_ref:
           Keyword.get(opts, :serving_ref, "serving:#{System.unique_integer([:positive])}"),
         model_ref: model_ref,
         surface: surface,
         compiled_taps: compiled_taps,
         predict_fun: predict_fun,
         metadata: %{
           hook_names: compiled_taps.hook_names,
           lifecycle: [:plan_compilation, :serving_compilation]
         }
       }}
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

  defp compiled_tap_summary(compiled_taps) do
    %{
      tap_plan_id: compiled_taps.tap_plan_id,
      global_layer_options: Map.new(compiled_taps.global_layer_options),
      hook_names: compiled_taps.hook_names,
      matched_count: length(compiled_taps.matched),
      unsupported_optional_count: length(compiled_taps.unsupported_optional)
    }
  end
end
