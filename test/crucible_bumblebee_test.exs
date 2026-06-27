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

  test "tap compiler maps captured outputs including deep fork activations" do
    plan =
      TapPlan.new!(
        [
          [id: "hidden", signal_type: :middle_residuals, layers: [0]],
          [id: "attention-pattern", signal_type: :attention_weights, layers: [0]],
          [id: "attention-scores", signal_type: :attention_scores, layers: [0]],
          [id: "attention-q", signal_type: :attention_q, layers: [0], required?: false],
          [id: "mlp-gate", signal_type: :mlp_gates, layers: [0], required?: false],
          [id: "final-norm", signal_type: :norm_telemetry, layers: [:final]],
          [id: "cache-state", signal_type: :kv_cache_state, required?: false],
          [id: "logits", signal_type: :final_logits, layers: [:final]]
        ],
        plan_id: "tap-plan-1"
      )

    assert {:ok, compiled} = TapCompiler.compile(plan, ExampleSurface.surface(num_blocks: 1))
    assert compiled.global_layer_options[:output_hidden_states]
    assert compiled.global_layer_options[:output_attentions]
    assert compiled.global_layer_options[:output_attention_qkv]
    assert compiled.global_layer_options[:output_attention_scores]
    assert compiled.global_layer_options[:output_mlp_activations]
    assert compiled.global_layer_options[:output_norm_telemetry]
    assert "decoder.layers.0.attention.weights" in compiled.hook_names
    assert "decoder.layers.0.attention.scores" in compiled.hook_names
    assert "decoder.layers.0.attention.query" in compiled.hook_names
    assert "decoder.layers.0.mlp.gate" in compiled.hook_names
    assert "outputs.norm_scales" in compiled.hook_names
    assert "outputs.norm_normalized" in compiled.hook_names
    assert "lm_head.output" in compiled.hook_names
    unsupported = Map.new(compiled.unsupported_optional, &{&1.tap_id, &1.reason})
    assert unsupported["cache-state"] == :no_surface_node
    assert compiled.metadata.surface_id == :example_transformer
  end

  test "required QKV taps compile against the deep Bumblebee fork surface" do
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

    assert {:ok, compiled} = TapCompiler.compile(plan, Qwen3Surface.surface(num_blocks: 1))
    assert compiled.global_layer_options[:output_attention_qkv]
    assert "decoder.blocks.0.self_attention.query" in compiled.hook_names
  end

  test "activation-name taps derive precise fork output options" do
    surface = ExampleSurface.surface(num_blocks: 1)

    assert {:ok, scores} =
             "attn-scores"
             |> CrucibleTap.activation_tap("blocks.0.attn.hook_attn_scores")
             |> TapCompiler.compile(surface)

    assert scores.global_layer_options[:output_attention_scores]
    refute scores.global_layer_options[:output_attention_qkv]

    assert {:ok, attn_out} =
             "attn-out"
             |> CrucibleTap.activation_tap("blocks.0.hook_attn_out")
             |> TapCompiler.compile(surface)

    assert attn_out.global_layer_options[:output_attention_qkv]
    refute attn_out.global_layer_options[:output_residual_streams]

    assert {:ok, mlp_out} =
             "mlp-out"
             |> CrucibleTap.activation_tap("blocks.0.hook_mlp_out")
             |> TapCompiler.compile(surface)

    assert mlp_out.global_layer_options[:output_mlp_activations]
    refute mlp_out.global_layer_options[:output_residual_streams]

    assert {:ok, resid_mid} =
             "resid-mid"
             |> CrucibleTap.activation_tap("blocks.0.hook_resid_mid")
             |> TapCompiler.compile(surface)

    assert resid_mid.global_layer_options[:output_residual_streams]

    assert {:ok, norm_scale} =
             "norm-scale"
             |> CrucibleTap.activation_tap("ln_final.hook_scale")
             |> TapCompiler.compile(surface)

    assert norm_scale.global_layer_options[:output_norm_telemetry]
    refute norm_scale.global_layer_options[:output_residual_streams]
    assert norm_scale.hook_names == ["outputs.norm_scales"]
  end

  test "signal extractor builds records and layer trajectory from fixture outputs" do
    {records, trajectory} =
      fixture_outputs()
      |> SignalExtractor.extract(trace_id: "trace-1", model_id: "fixture")

    signal_types = Enum.map(records, & &1.signal_type)

    assert :final_logits in signal_types
    assert :embeddings in signal_types
    assert :attention_weights in signal_types
    assert :attention_q in signal_types
    assert :attention_k in signal_types
    assert :attention_v in signal_types
    assert :attention_scores in signal_types
    assert :mlp_gates in signal_types
    assert :mlp_activation in signal_types
    assert :residual_stream in signal_types
    assert :norm_telemetry in signal_types
    assert trajectory.points != []

    final_logits = Enum.find(records, &(&1.signal_type == :final_logits))
    assert final_logits.metadata.activation_name == "unembed.hook_logits"
    assert final_logits.node_name == "final_logits"
    assert final_logits.capture_method == :bumblebee_output

    hidden = Enum.find(records, &(&1.signal_id == "hidden_states:0"))
    assert hidden.metadata.activation_name == "blocks.0.hook_resid_pre"

    attention = Enum.find(records, &(&1.signal_type == :attention_weights))
    assert attention.metadata.activation_name == "blocks.0.attn.hook_pattern"

    attention_q = Enum.find(records, &(&1.signal_type == :attention_q))
    assert attention_q.metadata.activation_name == "blocks.0.attn.hook_q"

    attention_scores = Enum.find(records, &(&1.signal_type == :attention_scores))
    assert attention_scores.metadata.activation_name == "blocks.0.attn.hook_attn_scores"

    mlp_pre = Enum.find(records, &(&1.signal_type == :mlp_gates))
    assert mlp_pre.metadata.activation_name == "blocks.0.mlp.hook_pre"

    resid_mid =
      Enum.find(records, &(&1.metadata.activation_name == "blocks.0.hook_resid_mid"))

    assert resid_mid.signal_type == :residual_stream

    norm_scale = Enum.find(records, &(&1.signal_id == "norm_scales"))
    assert norm_scale.metadata.activation_name == "ln_final.hook_scale"
    assert norm_scale.layer_index == :final

    norm_normalized = Enum.find(records, &(&1.signal_id == "norm_normalized"))
    assert norm_normalized.metadata.activation_name == "ln_final.hook_normalized"
    assert norm_normalized.layer_index == :final
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
      attention_queries: {Nx.tensor([[[[1.0, 0.0]]]], type: :f32)},
      attention_keys: {Nx.tensor([[[[0.8, 0.2]]]], type: :f32)},
      attention_values: {Nx.tensor([[[[0.3, 0.7]]]], type: :f32)},
      attention_scores: {Nx.tensor([[[[0.0]]]], type: :f32)},
      attention_zs: {Nx.tensor([[[[0.4, 0.6]]]], type: :f32)},
      attention_outputs: {Nx.tensor([[[0.4, 0.6]]], type: :f32)},
      mlp_pre_activations: {Nx.tensor([[[0.1, 0.2, 0.3]]], type: :f32)},
      mlp_post_activations: {Nx.tensor([[[0.2, 0.3, 0.4]]], type: :f32)},
      mlp_outputs: {Nx.tensor([[[0.9, 0.1]]], type: :f32)},
      residual_streams_pre: {Nx.tensor([[[1.0, 0.0]]], type: :f32)},
      residual_streams_mid: {Nx.tensor([[[0.5, 0.5]]], type: :f32)},
      residual_streams_post: {Nx.tensor([[[0.0, 1.0]]], type: :f32)},
      norm_scales: Nx.tensor([[[1.0]]], type: :f32),
      norm_normalized: Nx.tensor([[[0.0, 1.0]]], type: :f32),
      cache: %{blocks: {:block0}}
    }
  end
end
