defmodule CrucibleBumblebee.LogitLensRunnerTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{ExampleSurface, ForwardRunner, LogitLensRunner}
  alias CrucibleTap.TapPlan

  test "projects raw Bumblebee output residuals through surface logit-lens params" do
    params = %{
      decoder: %{final_norm: fn x -> x end},
      lm_head: %{kernel: Nx.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])}
    }

    assert {:ok, {logits, labels}} =
             LogitLensRunner.project_outputs(
               fixture_outputs(),
               ExampleSurface.surface(num_blocks: 2),
               %{n_layers: 2},
               params
             )

    assert labels == ["0_pre", "1_pre", "final_post"]
    assert Nx.shape(logits) == {3, 1, 3}
    assert Nx.to_flat_list(logits) == [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0]
  end

  test "emits bounded intermediate-logit signal records from raw outputs" do
    params = %{
      decoder: %{final_norm: fn x -> x end},
      lm_head: %{kernel: Nx.tensor([[1.0, 0.0], [0.0, 1.0]])}
    }

    assert {:ok, records} =
             LogitLensRunner.intermediate_records_from_outputs(
               fixture_outputs(),
               %{
                 trace_id: "trace-logit-lens",
                 model_id: "fixture",
                 model_family: :example_transformer,
                 backend: :binary
               },
               ExampleSurface.surface(num_blocks: 2),
               %{n_layers: 2},
               params
             )

    assert Enum.map(records, & &1.signal_type) == [
             :logit_lens_intermediate,
             :logit_lens_intermediate,
             :logit_lens_intermediate
           ]

    assert hd(records).metadata.logit_lens_label == "0_pre"
    assert hd(records).tensor_summary.shape == [1, 2]
  end

  test "summary-only traces do not masquerade as logit-lens inputs" do
    params = %{
      decoder: %{final_norm: fn x -> x end},
      lm_head: %{kernel: Nx.tensor([[1.0, 0.0], [0.0, 1.0]])}
    }

    assert {:error, {:raw_activations_required, "blocks.0.hook_resid_pre"}} =
             LogitLensRunner.project_trace(
               fixture_trace!(),
               ExampleSurface.surface(num_blocks: 2),
               %{n_layers: 2},
               params
             )
  end

  defp fixture_outputs do
    %{
      logits: Nx.tensor([[0.1, 0.2]], type: :f32),
      hidden_states: {
        Nx.tensor([[1.0, 0.0]], type: :f32),
        Nx.tensor([[0.0, 1.0]], type: :f32),
        Nx.tensor([[1.0, 1.0]], type: :f32)
      },
      attentions: {Nx.tensor([[[[1.0]]]], type: :f32)}
    }
  end

  defp fixture_trace! do
    plan =
      TapPlan.new!(
        [
          [id: "hidden", signal_type: :middle_residuals, layers: [0]],
          [id: "logits", signal_type: :final_logits, layers: [:final]]
        ],
        plan_id: "tap-plan-logit-lens"
      )

    predict_fun = fn _inputs -> fixture_outputs() end

    {:ok, trace} =
      ForwardRunner.run(predict_fun, %{}, plan,
        trace_id: "trace-logit-lens",
        model_id: "fixture",
        surface: ExampleSurface.surface(num_blocks: 2)
      )

    trace
  end
end
