defmodule CrucibleBumblebee.ArtifactsTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.Artifacts

  setup do
    root =
      Path.join(
        System.tmp_dir!(),
        "crucible_bumblebee_artifacts_#{System.unique_integer([:positive])}"
      )

    on_exit(fn ->
      File.rm_rf!(root)
    end)

    {:ok, root: root}
  end

  test "ensures the artifact directory layout", %{root: root} do
    assert Artifacts.ensure_layout!(root: root) == root

    for directory <- Artifacts.directories() |> Map.values() do
      assert File.dir?(Path.join(root, directory))
    end
  end

  test "builds stable native trace and capability report paths", %{root: root} do
    assert Artifacts.trace_path("gpt2 binary", root: root) ==
             Path.join([root, "traces/native", "gpt2_binary.trace.jsonl"])

    assert Artifacts.capability_report_path("gpt2 binary", root: root) ==
             Path.join([root, "capability_reports", "gpt2_binary.capability_report.json"])
  end

  test "writes JSONL matrix rows and artifact index entries", %{root: root} do
    row_path =
      Artifacts.append_jsonl!(:model_matrix, "gpt2_binary.jsonl", %{model_id: "gpt2"}, root: root)

    index_path =
      Artifacts.append_index!(%{phase: "2", command: "mix ci", exit_code: 0}, root: root)

    assert row_path == Path.join([root, "model_matrix", "gpt2_binary.jsonl"])
    assert File.read!(row_path) == ~s({"model_id":"gpt2"}\n)

    assert index_path == Path.join(root, "ARTIFACT_INDEX.md")
    assert File.read!(index_path) =~ "| 2 | mix ci |"
  end
end
