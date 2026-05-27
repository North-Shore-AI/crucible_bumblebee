defmodule CrucibleBumblebee.TraceWriter do
  @moduledoc """
  V4 trace writer for native Bumblebee examples.
  """

  alias CrucibleBumblebee.Artifacts
  alias CrucibleSignalTrace.JSONL

  def trace_dir do
    System.get_env("CRUCIBLE_TRACE_DIR") || Artifacts.dir!(:native_traces)
  end

  def output_path(name, suffix) do
    if System.get_env("CRUCIBLE_TRACE_DIR") do
      Path.join(trace_dir(), "#{Artifacts.safe_name(name)}.#{suffix}")
    else
      case suffix do
        "trace.jsonl" ->
          Artifacts.trace_path(name)

        "capability_report.json" ->
          Artifacts.capability_report_path(name)

        other ->
          Path.join(trace_dir(), "#{Artifacts.safe_name(name)}.#{other}")
      end
    end
  end

  def reset!(path) do
    File.rm(path)
    File.mkdir_p!(Path.dirname(path))
    :ok
  end

  def write!(path, event_type, attrs) do
    JSONL.write_event!(path, JSONL.v4_event(event_type, attrs))
  end

  def write_capability_report!(path, report) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Crucible.CanonicalJSON.encode!(report) <> "\n")
    path
  end

  def signal_from_logits(logits, attrs) do
    %Crucible.SignalRecord{
      signal_id: Map.fetch!(attrs, :signal_id),
      trace_id: Map.fetch!(attrs, :trace_id),
      run_id: Map.fetch!(attrs, :run_id),
      signal_type: :final_logits,
      provider_kind: :elixir_bumblebee,
      model_id: Map.fetch!(attrs, :model_id),
      model_family: Map.fetch!(attrs, :model_family),
      backend: Map.fetch!(attrs, :backend),
      layer_index: nil,
      token_index: nil,
      node_name: "final_logits",
      capture_method: :axon_hook,
      tensor_summary: Crucible.TensorSummary.compute(logits, entropy: true, top_k: 10),
      tensor_ref: nil,
      metadata: Map.get(attrs, :metadata, %{})
    }
  end
end
