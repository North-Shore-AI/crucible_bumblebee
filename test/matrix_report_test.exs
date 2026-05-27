defmodule CrucibleBumblebee.MatrixReportTest do
  use ExUnit.Case, async: false

  alias CrucibleBumblebee.{Artifacts, MatrixReport}

  test "writes latest matrix rows to markdown reports" do
    root =
      Path.join(
        System.tmp_dir!(),
        "crucible_bumblebee_matrix_report_#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf!(root) end)

    Artifacts.ensure_layout!(root: root)

    Artifacts.append_jsonl!(
      :model_matrix,
      "model_ladder.jsonl",
      %{rung: "M1", model_id: "gpt2", backend: :binary, result: "failed"},
      root: root
    )

    Artifacts.append_jsonl!(
      :model_matrix,
      "model_ladder.jsonl",
      %{rung: "M1", model_id: "gpt2", backend: :binary, result: "passed"},
      root: root
    )

    reports = MatrixReport.write_from_artifacts!(artifact_root: root)
    assert {:model_matrix, model_report} = List.keyfind(reports, :model_matrix, 0)
    text = File.read!(model_report)

    assert text =~ "# Model Ladder Matrix"
    assert text =~ "| M1 | gpt2 |"
    assert text =~ "passed"
    refute text =~ "failed"
  end
end
