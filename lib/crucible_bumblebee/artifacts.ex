defmodule CrucibleBumblebee.Artifacts do
  @moduledoc """
  Shared V5 artifact path and index helpers for native Bumblebee execution.
  """

  @default_root "tmp/crucible_v5"

  @directories %{
    transcripts: "transcripts",
    native_traces: "traces/native",
    synthetic_traces: "traces/synthetic",
    capability_reports: "capability_reports",
    policy_decisions: "policy_decisions",
    route_decisions: "route_decisions",
    model_matrix: "model_matrix",
    backend_matrix: "backend_matrix",
    signal_matrix: "signal_matrix",
    generation_matrix: "generation_matrix",
    internals_matrix: "internals_matrix",
    errors: "errors",
    graphs: "graphs",
    reports: "reports",
    git: "git"
  }

  @type directory ::
          :transcripts
          | :native_traces
          | :synthetic_traces
          | :capability_reports
          | :policy_decisions
          | :route_decisions
          | :model_matrix
          | :backend_matrix
          | :signal_matrix
          | :generation_matrix
          | :internals_matrix
          | :errors
          | :graphs
          | :reports
          | :git

  @spec root(keyword()) :: Path.t()
  def root(opts \\ []) do
    Keyword.get(opts, :root) ||
      System.get_env("CRUCIBLE_V5_ARTIFACT_ROOT") ||
      System.get_env("CRUCIBLE_ARTIFACT_ROOT") ||
      @default_root
  end

  @spec directories() :: %{directory() => Path.t()}
  def directories, do: @directories

  @spec ensure_layout!(keyword()) :: Path.t()
  def ensure_layout!(opts \\ []) do
    root = root(opts)

    @directories
    |> Map.values()
    |> Enum.each(fn directory -> File.mkdir_p!(Path.join(root, directory)) end)

    root
  end

  @spec dir!(directory(), keyword()) :: Path.t()
  def dir!(directory, opts \\ []) do
    child = Map.fetch!(@directories, directory)
    Path.join(root(opts), child)
  end

  @spec path!(directory(), String.t(), keyword()) :: Path.t()
  def path!(directory, filename, opts \\ []) do
    directory
    |> dir!(opts)
    |> Path.join(filename)
  end

  @spec trace_path(String.t(), keyword()) :: Path.t()
  def trace_path(name, opts \\ []) do
    path!(:native_traces, "#{safe_name(name)}.trace.jsonl", opts)
  end

  @spec capability_report_path(String.t(), keyword()) :: Path.t()
  def capability_report_path(name, opts \\ []) do
    path!(:capability_reports, "#{safe_name(name)}.capability_report.json", opts)
  end

  @spec matrix_row_path(directory(), String.t(), keyword()) :: Path.t()
  def matrix_row_path(directory, name, opts \\ [])
      when directory in [
             :model_matrix,
             :backend_matrix,
             :signal_matrix,
             :generation_matrix,
             :internals_matrix
           ] do
    path!(directory, "#{safe_name(name)}.jsonl", opts)
  end

  @spec index_path(keyword()) :: Path.t()
  def index_path(opts \\ []), do: Path.join(root(opts), "ARTIFACT_INDEX.md")

  @spec ensure_index!(keyword()) :: Path.t()
  def ensure_index!(opts \\ []) do
    ensure_layout!(opts)
    path = index_path(opts)

    unless File.exists?(path) do
      File.write!(path, index_header())
    end

    path
  end

  @spec append_index!(map(), keyword()) :: Path.t()
  def append_index!(entry, opts \\ []) when is_map(entry) do
    path = ensure_index!(opts)
    File.write!(path, index_row(entry), [:append])
    path
  end

  @spec write_json!(directory(), String.t(), map(), keyword()) :: Path.t()
  def write_json!(directory, filename, payload, opts \\ []) when is_map(payload) do
    ensure_layout!(opts)
    path = path!(directory, filename, opts)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(payload, pretty: true) <> "\n")
    path
  end

  @spec append_jsonl!(directory(), String.t(), map(), keyword()) :: Path.t()
  def append_jsonl!(directory, filename, payload, opts \\ []) when is_map(payload) do
    ensure_layout!(opts)
    path = path!(directory, filename, opts)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(payload) <> "\n", [:append])
    path
  end

  @spec safe_name(String.t() | atom()) :: String.t()
  def safe_name(name) when is_atom(name), do: safe_name(Atom.to_string(name))

  def safe_name(name) when is_binary(name) do
    name
    |> String.trim()
    |> String.replace(~r/[^A-Za-z0-9_.-]+/, "_")
    |> String.trim("_")
    |> case do
      "" -> "artifact"
      safe -> safe
    end
  end

  defp index_header do
    """
    # Crucible V5 Artifact Index

    | Phase | Command | CWD | Env | Exit | Transcript | Artifacts | Git commit |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    """
  end

  defp index_row(entry) do
    [
      Map.get(entry, :phase, Map.get(entry, "phase", "")),
      Map.get(entry, :command, Map.get(entry, "command", "")),
      Map.get(entry, :cwd, Map.get(entry, "cwd", "")),
      Map.get(entry, :env, Map.get(entry, "env", "")),
      Map.get(entry, :exit_code, Map.get(entry, "exit_code", "")),
      Map.get(entry, :transcript, Map.get(entry, "transcript", "")),
      Map.get(entry, :artifacts, Map.get(entry, "artifacts", "")),
      Map.get(entry, :git_commit, Map.get(entry, "git_commit", ""))
    ]
    |> Enum.map(&markdown_cell/1)
    |> then(&("| " <> Enum.join(&1, " | ") <> " |\n"))
  end

  defp markdown_cell(value) when is_list(value), do: value |> Enum.join("<br>") |> markdown_cell()
  defp markdown_cell(value), do: value |> to_string() |> String.replace("|", "\\|")
end
