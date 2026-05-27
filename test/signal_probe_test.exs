defmodule CrucibleBumblebee.SignalProbeTest do
  use ExUnit.Case, async: false

  alias CrucibleBumblebee.SignalProbe

  test "probe failure produces a structured signal matrix row" do
    model = %{
      rung: "Mbad",
      model_id: "unsupported/model",
      family: :unknown,
      architecture: :for_causal_language_modeling
    }

    root =
      Path.join(
        System.tmp_dir!(),
        "crucible_bumblebee_signal_probe_#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf!(root) end)

    result = SignalProbe.run_model(model, backend: :binary, artifact_root: root)

    refute result.ok
    text = File.read!(Path.join([root, "signal_matrix", "signal_probe.jsonl"]))
    assert text =~ "failed_with_exception"
    assert text =~ "unsupported/model"
  end
end
