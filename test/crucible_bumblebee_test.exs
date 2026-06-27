defmodule CrucibleBumblebeeTest do
  use ExUnit.Case
  doctest CrucibleBumblebee

  alias CrucibleBumblebee.{
    CacheSummary,
    ExampleSurface,
    ForwardRunner,
    LogitsProcessor,
    Qwen3Surface,
    SignalExtractor,
    TapCompiler
  }

  alias CruciblePolicy.SteeringPlan
  alias Crucible.ForwardTrace
  alias CrucibleTap.TapPlan

  test "exposes package version" do
    assert CrucibleBumblebee.version() == "0.1.0"
  end

  test "example surface exposes expected generic layer names" do
    surface = ExampleSurface.surface(num_blocks: 2)
    names = Enum.map(surface.surface.nodes, & &1.layer_name)

    assert "tokens.embedding" in names
    assert "decoder.layers.0.attention.query" in names
    assert "decoder.layers.1.mlp.gate" in names
    assert "lm_head.output" in names

    q_node = Enum.find(surface.surface.nodes, &(&1.id == "decoder.layers.0.attention.query"))
    assert q_node.activation_name == "blocks.0.attn.hook_q"
    assert q_node.axes == [:batch, :pos, :head, :d_head]
  end

  test "qwen3 example surface exposes qwen-family layer names without becoming the default" do
    surface = Qwen3Surface.surface(num_blocks: 2)
    names = Enum.map(surface.surface.nodes, & &1.layer_name)

    assert "embedder.token_embedding" in names
    assert "decoder.blocks.0.self_attention.query" in names
    assert "decoder.blocks.1.ffn.gate" in names
    assert "language_modeling_head.output" in names
  end

  test "tap compiler maps captured outputs and degrades unavailable deep hooks" do
    plan =
      TapPlan.new!(
        [
          [id: "hidden", signal_type: :middle_residuals, layers: [0]],
          [id: "attention-pattern", signal_type: :attention_weights, layers: [0]],
          [id: "attention-q", signal_type: :attention_q, layers: [0], required?: false],
          [id: "cache-state", signal_type: :kv_cache_state, required?: false],
          [id: "logits", signal_type: :final_logits, layers: [:final]]
        ],
        plan_id: "tap-plan-1"
      )

    assert {:ok, compiled} = TapCompiler.compile(plan, ExampleSurface.surface(num_blocks: 1))
    assert compiled.global_layer_options[:output_hidden_states]
    assert compiled.global_layer_options[:output_attentions]
    assert "decoder.layers.0.attention.weights" in compiled.hook_names
    assert "lm_head.output" in compiled.hook_names
    unsupported = Map.new(compiled.unsupported_optional, &{&1.tap_id, &1.reason})
    assert unsupported["attention-q"] == :unsupported_operation
    assert unsupported["cache-state"] == :no_surface_node
    assert compiled.metadata.surface_id == :example_transformer
  end

  test "required QKV taps fail closed before deep Bumblebee instrumentation exists" do
    plan =
      TapPlan.new!(
        [
          [
            id: "q-required",
            signal_type: :attention_q,
            layers: [0],
            required?: true
          ]
        ],
        plan_id: "tap-plan-required-q"
      )

    assert {:error, report} = TapCompiler.compile(plan, Qwen3Surface.surface(num_blocks: 1))
    assert [%{tap_id: "q-required", reason: :unsupported_operation}] = report.unsupported_required
  end

  test "signal extractor builds records and layer trajectory from fixture outputs" do
    {records, trajectory} =
      fixture_outputs()
      |> SignalExtractor.extract(trace_id: "trace-1", model_id: "fixture")

    signal_types = Enum.map(records, & &1.signal_type)

    assert :final_logits in signal_types
    assert :embeddings in signal_types
    assert :attention_weights in signal_types
    assert trajectory.points != []

    final_logits = Enum.find(records, &(&1.signal_type == :final_logits))
    assert final_logits.metadata.activation_name == "unembed.hook_logits"
    assert final_logits.node_name == "final_logits"
    assert final_logits.capture_method == :bumblebee_output

    hidden = Enum.find(records, &(&1.signal_id == "hidden_states:0"))
    assert hidden.metadata.activation_name == "blocks.0.hook_resid_pre"

    attention = Enum.find(records, &(&1.signal_type == :attention_weights))
    assert attention.metadata.activation_name == "blocks.0.attn.hook_pattern"
  end

  test "forward runner returns a forward trace from a tiny fixture predict function" do
    plan =
      TapPlan.new!(
        [
          [id: "hidden", signal_type: :middle_residuals, layers: [0]],
          [id: "logits", signal_type: :final_logits, layers: [:final]]
        ],
        plan_id: "tap-plan-run"
      )

    predict_fun = fn _inputs -> fixture_outputs() end

    assert {:ok, %ForwardTrace{} = trace} =
             ForwardRunner.run(predict_fun, %{}, plan,
               trace_id: "trace-run",
               model_id: "fixture",
               surface: ExampleSurface.surface(num_blocks: 1)
             )

    assert trace.trace_id == "trace-run"
    assert trace.final_logits.signal_type == :final_logits
    assert trace.final_logits.metadata.activation_name == "unembed.hook_logits"
    assert trace.cache_summary.blocks == 1
    assert trace.metadata.lifecycle == [:plan_compilation, :serving_compilation, :execution]
  end

  test "forward runner fails closed when compiled output options are absent" do
    plan =
      TapPlan.new!(
        [
          [id: "hidden", signal_type: :middle_residuals, layers: [0], required?: true],
          [id: "logits", signal_type: :final_logits, layers: [:final]]
        ],
        plan_id: "tap-plan-missing-hidden"
      )

    predict_fun = fn _inputs -> %{logits: Nx.tensor([[0.1, 0.3, 0.2]], type: :f32)} end

    assert {:error, {:missing_compiled_output, :hidden_states}} =
             ForwardRunner.run(predict_fun, %{}, plan,
               trace_id: "trace-run-missing-hidden",
               model_id: "fixture",
               surface: ExampleSurface.surface(num_blocks: 1)
             )
  end

  test "logits processor applies policy steering plan" do
    plan =
      SteeringPlan.new!(trace_id: "trace-1", token_biases: %{1 => 2.0}, banned_token_ids: [0])

    assert LogitsProcessor.process([1.0, 1.0], plan) == [:neg_infinity, 3.0]
  end

  test "cache summaries are bounded" do
    assert CacheSummary.summarize(%{blocks: {:a, :b}}) == %{keys: ["blocks"], blocks: 2}
    assert CacheSummary.summarize({:a, :b, :c}) == %{tuple_size: 3}
  end

  defp fixture_outputs do
    %{
      logits: Nx.tensor([[0.1, 0.3, 0.2]], type: :f32),
      hidden_states: {
        Nx.tensor([[1.0, 0.0]], type: :f32),
        Nx.tensor([[0.0, 1.0]], type: :f32),
        Nx.tensor([[1.0, 1.0]], type: :f32)
      },
      attentions: {Nx.tensor([[0.5, 0.5]], type: :f32)},
      cache: %{blocks: {:block0}}
    }
  end
end
