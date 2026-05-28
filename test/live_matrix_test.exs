defmodule CrucibleBumblebee.LiveMatrixTest do
  use ExUnit.Case, async: false

  alias CrucibleBumblebee.LiveMatrix

  test "model ladder names all required rungs" do
    assert LiveMatrix.model_ladder() |> Enum.map(& &1.rung) == ~w(M0 M1 M2 M3 M4 M5 M6)
    assert Enum.find(LiveMatrix.model_ladder(), &(&1.model_id == "gpt2"))
    assert Enum.find(LiveMatrix.model_ladder(), &(&1.family == :qwen3))
  end

  test "model ladder writes model, backend, and signal matrix rows" do
    root =
      Path.join(
        System.tmp_dir!(),
        "crucible_bumblebee_live_matrix_#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf!(root) end)

    models = [%{rung: "Mtest", model_id: "fixture/model", family: :fixture}]

    result =
      LiveMatrix.run_model_ladder(
        artifact_root: root,
        backend: :binary,
        models: models,
        attempt_fun: fn model, backend, _root ->
          %{
            rung: model.rung,
            model_id: model.model_id,
            family: model.family,
            backend: backend,
            tokenizer_loaded: true,
            model_loaded: true,
            forward: true,
            final_logits: true,
            generation: false,
            step_logits: false,
            hidden_states: false,
            attention: false,
            trace: "trace.jsonl",
            policy_replay: false,
            trinity_trace: false,
            trinity_live: false,
            duration_ms: 1,
            result: "passed",
            error: nil
          }
        end
      )

    assert result.ok
    assert File.read!(Path.join([root, "model_matrix", "model_ladder.jsonl"])) =~ "fixture/model"
    assert File.read!(Path.join([root, "backend_matrix", "backend_ladder.jsonl"])) =~ "binary"
    assert File.read!(Path.join([root, "signal_matrix", "signal_ladder.jsonl"])) =~ "final_logits"
  end

  test "generation ladder writes generation matrix rows" do
    root =
      Path.join(
        System.tmp_dir!(),
        "crucible_bumblebee_generation_matrix_#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf!(root) end)

    models = [%{rung: "Mtest", model_id: "fixture/model", family: :fixture}]

    result =
      LiveMatrix.run_generation_ladder(
        artifact_root: root,
        backend: :binary,
        models: models,
        attempt_fun: fn model, backend, generation_opts, _root ->
          %{
            rung: model.rung,
            model_id: model.model_id,
            family: model.family,
            backend: backend,
            generation: true,
            success_level: :generation_step_logits,
            generated_tokens: [1],
            step_logits: true,
            step_count: Keyword.fetch!(generation_opts, :max_new_tokens),
            trace: "trace.jsonl",
            duration_ms: 1,
            result: "passed",
            error: nil
          }
        end
      )

    assert result.ok

    assert File.read!(Path.join([root, "generation_matrix", "generation_ladder.jsonl"])) =~
             "generation_step_logits"
  end

  test "backend ladder records unavailable backend rows" do
    root =
      Path.join(
        System.tmp_dir!(),
        "crucible_bumblebee_backend_matrix_#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf!(root) end)

    models = [%{rung: "Mtest", model_id: "fixture/model", family: :fixture}]

    result =
      LiveMatrix.run_backend_ladder(
        artifact_root: root,
        models: models,
        backends: [:exla_cpu],
        backend_probe_fun: fn :exla_cpu -> {:error, {:not_installed, :exla}} end
      )

    assert result.ok

    text = File.read!(Path.join([root, "backend_matrix", "backend_ladder.jsonl"]))
    assert text =~ "backend_unavailable"
    assert text =~ "not_installed"
  end
end
