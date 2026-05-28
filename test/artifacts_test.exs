defmodule CrucibleBumblebee.ArtifactsTest do
  use ExUnit.Case, async: false

  alias CrucibleBumblebee.Artifacts

  setup do
    previous_root = System.get_env("CRUCIBLE_ARTIFACT_ROOT")

    root =
      Path.join(
        System.tmp_dir!(),
        "crucible_bumblebee_artifacts_#{System.unique_integer([:positive])}"
      )

    System.put_env("CRUCIBLE_ARTIFACT_ROOT", root)

    on_exit(fn ->
      restore_env("CRUCIBLE_ARTIFACT_ROOT", previous_root)
      File.rm_rf!(root)
    end)

    {:ok, root: root}
  end

  test "ensures the artifact directory layout", %{root: root} do
    assert Artifacts.ensure_layout!() == root

    for directory <- Artifacts.directories() |> Map.values() do
      assert File.dir?(Path.join(root, directory))
    end
  end

  test "builds stable native trace and capability report paths", %{root: root} do
    assert Artifacts.trace_path("gpt2 binary") ==
             Path.join([root, "traces/native", "gpt2_binary.trace.jsonl"])

    assert Artifacts.capability_report_path("gpt2 binary") ==
             Path.join([root, "capability_reports", "gpt2_binary.capability_report.json"])
  end

  test "writes JSONL matrix rows and artifact index entries", %{root: root} do
    row_path = Artifacts.append_jsonl!(:model_matrix, "gpt2_binary.jsonl", %{model_id: "gpt2"})
    index_path = Artifacts.append_index!(%{phase: "2", command: "mix ci", exit_code: 0})

    assert row_path == Path.join([root, "model_matrix", "gpt2_binary.jsonl"])
    assert File.read!(row_path) == ~s({"model_id":"gpt2"}\n)

    assert index_path == Path.join(root, "ARTIFACT_INDEX.md")
    assert File.read!(index_path) =~ "| 2 | mix ci |"
  end

  defp restore_env(key, nil), do: System.delete_env(key)
  defp restore_env(key, value), do: System.put_env(key, value)
end
