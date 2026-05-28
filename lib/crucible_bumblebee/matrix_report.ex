defmodule CrucibleBumblebee.MatrixReport do
  @moduledoc """
  Writes human-readable V5 matrix reports from JSONL artifact rows.
  """

  alias CrucibleBumblebee.Artifacts

  @reports [
    model_matrix: {"model_ladder.jsonl", "model_matrix.md", "Model Ladder Matrix"},
    backend_matrix: {"backend_ladder.jsonl", "backend_matrix.md", "Backend Ladder Matrix"},
    signal_matrix:
      {["signal_ladder.jsonl", "signal_probe.jsonl"], "signal_matrix.md", "Signal Ladder Matrix"},
    generation_matrix:
      {"generation_ladder.jsonl", "generation_matrix.md", "Generation Ladder Matrix"},
    internals_matrix: {"internals_ladder.jsonl", "internals_matrix.md", "Native Internals Matrix"}
  ]

  def write_from_artifacts!(opts \\ []) do
    root = Keyword.get(opts, :artifact_root) || Artifacts.root(root: Keyword.get(opts, :root))
    Artifacts.ensure_layout!(root: root)

    Enum.map(@reports, fn {directory, {sources, target, title}} ->
      rows =
        sources
        |> List.wrap()
        |> Enum.flat_map(&read_rows(Artifacts.path!(directory, &1, root: root)))
        |> latest_rows()

      path = Artifacts.path!(:reports, target, root: root)
      File.write!(path, markdown(title, rows))
      {directory, path}
    end)
  end

  def markdown(title, rows) when is_binary(title) and is_list(rows) do
    columns = columns(rows)

    [
      "# #{title}\n\n",
      table(columns, rows)
    ]
    |> IO.iodata_to_binary()
  end

  defp read_rows(path) do
    if File.exists?(path) do
      path
      |> File.stream!([], :line)
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Enum.map(&Jason.decode!/1)
    else
      []
    end
  end

  defp latest_rows(rows) do
    rows
    |> Enum.with_index()
    |> Enum.reduce(%{}, fn {row, index}, acc ->
      Map.put(acc, row_key(row), {index, row})
    end)
    |> Map.values()
    |> Enum.sort_by(fn {index, _row} -> index end)
    |> Enum.map(fn {_index, row} -> row end)
  end

  defp row_key(row) do
    [
      Map.get(row, "rung"),
      Map.get(row, "model_id"),
      Map.get(row, "backend"),
      Map.get(row, "signal"),
      Map.get(row, "step_count"),
      Map.get(row, "generation_strategy"),
      Map.get(row, "stop_token_ids")
    ]
    |> Enum.reject(&is_nil/1)
    |> Enum.join("|")
  end

  defp columns([]), do: ["result"]

  defp columns(rows) do
    preferred = [
      "rung",
      "model_id",
      "family",
      "backend",
      "result",
      "forward",
      "generation",
      "final_logits",
      "step_logits",
      "hidden_states",
      "attention",
      "success_level",
      "signal",
      "status",
      "step_count",
      "generation_strategy",
      "stop_token_ids",
      "generated_name",
      "op_name",
      "node_id",
      "node_count",
      "candidate_count",
      "duration_ms",
      "blocker",
      "error",
      "reason",
      "trace"
    ]

    present =
      rows
      |> Enum.flat_map(&Map.keys/1)
      |> Enum.uniq()

    preferred ++ Enum.sort(present -- preferred)
  end

  defp table(columns, rows) do
    [
      "| ",
      Enum.join(columns, " | "),
      " |\n| ",
      columns |> Enum.map(fn _ -> "---" end) |> Enum.join(" | "),
      " |\n",
      Enum.map(rows, fn row ->
        ["| ", columns |> Enum.map(&markdown_cell(Map.get(row, &1))) |> Enum.join(" | "), " |\n"]
      end)
    ]
  end

  defp markdown_cell(nil), do: ""

  defp markdown_cell(value) when is_list(value),
    do: value |> inspect(charlists: :as_lists) |> markdown_cell()

  defp markdown_cell(value) when is_map(value), do: value |> Jason.encode!() |> markdown_cell()

  defp markdown_cell(value) do
    value
    |> to_string()
    |> String.replace("|", "\\|")
    |> String.replace("\n", " ")
  end
end
