defmodule CrucibleBumblebee.InternalsProbeTest do
  use ExUnit.Case, async: false

  alias CrucibleBumblebee.{InternalsProbe, ModelBundle}

  test "run_bundle dumps graph metadata and captures passive hook summaries" do
    root =
      Path.join(
        System.tmp_dir!(),
        "crucible_bumblebee_internals_probe_#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf!(root) end)

    model =
      Axon.input("input", shape: {nil, 4})
      |> Axon.dense(3)
      |> Axon.dense(2)

    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())

    bundle = %ModelBundle{
      model_id: "fixture/model",
      model: model,
      params: params,
      backend: :binary,
      model_family: :fixture,
      spec: nil
    }

    result =
      InternalsProbe.run_bundle(
        bundle,
        Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32),
        %{rung: "Mfixture", model_id: "fixture/model", family: :fixture},
        artifact_root: root
      )

    assert result.ok
    assert File.exists?(result.graph_path)
    assert File.exists?(result.activation_cache_path)

    rows = File.read!(Path.join([root, "internals_matrix", "internals_ladder.jsonl"]))
    assert rows =~ "graph_dump"
    assert rows =~ "final_logits"
    assert rows =~ "mlp_activation"
    assert rows =~ "captured"

    report_paths = CrucibleBumblebee.MatrixReport.write_from_artifacts!(artifact_root: root)
    assert {:internals_matrix, report_path} = List.keyfind(report_paths, :internals_matrix, 0)
    assert File.read!(report_path) =~ "Native Internals Matrix"
  end
end
