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
  end

  test "qwen3 example surface exposes qwen-family layer names without becoming the default" do
    surface = Qwen3Surface.surface(num_blocks: 2)
    names = Enum.map(surface.surface.nodes, & &1.layer_name)

    assert "embedder.token_embedding" in names
    assert "decoder.blocks.0.self_attention.query" in names
    assert "decoder.blocks.1.ffn.gate" in names
    assert "language_modeling_head.output" in names
  end

  test "tap compiler maps hidden states, attentions, logits, cache-adjacent hooks, and named hooks" do
    plan =
      TapPlan.new!(
        [
          [id: "hidden", signal_type: :middle_residuals, layers: [0]],
          [id: "attention-q", signal_type: :attention_q, layers: [0]],
          [id: "cache-state", signal_type: :kv_cache_state, required?: false],
          [id: "logits", signal_type: :final_logits, layers: [:final]]
        ],
        plan_id: "tap-plan-1"
      )

    assert {:ok, compiled} = TapCompiler.compile(plan, ExampleSurface.surface(num_blocks: 1))
    assert compiled.global_layer_options[:output_hidden_states]
    assert compiled.global_layer_options[:output_attentions]
    assert "decoder.layers.0.attention.query" in compiled.hook_names
    assert "lm_head.output" in compiled.hook_names
    assert [%{tap_id: "cache-state"}] = compiled.unsupported_optional
    assert compiled.metadata.surface_id == :example_transformer
  end

  test "signal extractor builds records and layer trajectory from fixture outputs" do
    {records, trajectory} =
      fixture_outputs()
      |> SignalExtractor.extract(trace_id: "trace-1", model_id: "fixture")

    signal_types = Enum.map(records, & &1.signal_type)

    assert :final_logits in signal_types
    assert :embeddings in signal_types
    assert :attention_maps in signal_types
    assert trajectory.points != []
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
    assert trace.cache_summary.blocks == 1
    assert trace.metadata.lifecycle == [:plan_compilation, :serving_compilation, :execution]
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
