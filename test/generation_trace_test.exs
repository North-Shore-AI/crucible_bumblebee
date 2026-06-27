defmodule CrucibleBumblebee.GenerationTraceTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{GenerationTrace, Live, ManualGeneration, ModelBundle}

  test "cache-backed generation trace matches Bumblebee greedy generation and full-forward tokens" do
    bundle = tiny_gpt2_bundle()
    inputs = %{"input_ids" => Nx.tensor([[1, 2, 3]], type: :u32)}

    assert {:ok, trace} = GenerationTrace.run_inputs(bundle, inputs, max_new_tokens: 2, top_k: 3)

    assert trace.generation_success_level == :kv_cache_generation_trace
    assert trace.generated_token_ids == normal_bumblebee_tokens(bundle, inputs, 2)
    assert trace.generated_token_ids == full_forward_tokens(bundle, inputs, 2)
    assert Enum.map(trace.steps, & &1.cache_offset) == [3, 4]
    assert Enum.all?(trace.steps, &(&1.tensor_summary.top_k != []))

    assert Enum.all?(trace.steps, fn step ->
             [%{token_id: top_token_id} | _rest] = step.top_k
             top_token_id == step.token_id
           end)

    assert Enum.map(trace.steps, & &1.cache_metadata.prompt_length) == [3, 3]
    assert Enum.map(trace.steps, & &1.cache_metadata.generated_length) == [1, 2]

    assert Enum.map(trace.steps, & &1.cache_metadata.source) == [
             :bumblebee_generation_trace,
             :bumblebee_generation_trace
           ]
  end

  test "non-greedy strategies fail closed" do
    bundle = tiny_gpt2_bundle()
    inputs = %{"input_ids" => Nx.tensor([[1, 2, 3]], type: :u32)}

    assert {:error, {:unsupported_generation_trace_strategy, :top_k_sample}} =
             GenerationTrace.run_inputs(bundle, inputs,
               max_new_tokens: 1,
               strategy: :top_k_sample
             )
  end

  test "public step drops raw logits and keeps cache metadata" do
    step = %{
      step_index: 1,
      token_id: 3,
      logits: Nx.tensor([1.0, 2.0]),
      tensor_summary: Crucible.TensorSummary.compute(Nx.tensor([1.0, 2.0])),
      cache_metadata: %{offset: 4}
    }

    public = GenerationTrace.public_step(step)

    refute Map.has_key?(public, :logits)
    assert public.tensor_summary.rank == 1
    assert public.cache_metadata.offset == 4
  end

  defp tiny_gpt2_bundle do
    spec =
      Bumblebee.configure(Bumblebee.Text.Gpt2,
        architecture: :for_causal_language_modeling,
        vocab_size: 32,
        hidden_size: 4,
        num_blocks: 2,
        num_attention_heads: 2,
        max_positions: 8,
        intermediate_size: 8,
        dropout_rate: 0.0,
        embeddings_dropout_rate: 0.0,
        attention_dropout_rate: 0.0
      )

    model = Bumblebee.build_model(spec)
    {init_fn, _predict_fn} = Axon.build(model)
    template = %{"input_ids" => Nx.template({1, 3}, :u32)}
    params = init_fn.(template, Axon.ModelState.empty())

    %ModelBundle{
      model_id: "tiny-gpt2",
      model: model,
      params: params,
      spec: spec,
      tokenizer: nil
    }
  end

  defp full_forward_tokens(bundle, inputs, max_new_tokens) do
    1..max_new_tokens
    |> Enum.reduce({inputs, []}, fn _step, {step_inputs, tokens} ->
      outputs = Axon.predict(bundle.model, bundle.params, step_inputs)

      token_id =
        outputs
        |> Live.fetch_logits!()
        |> Live.last_token_logits()
        |> ManualGeneration.greedy_token_id()

      {ManualGeneration.append_token(step_inputs, token_id), [token_id | tokens]}
    end)
    |> elem(1)
    |> Enum.reverse()
  end

  defp normal_bumblebee_tokens(bundle, inputs, max_new_tokens) do
    generation_config =
      Bumblebee.configure(Bumblebee.Text.GenerationConfig,
        max_new_tokens: max_new_tokens,
        pad_token_id: 0,
        eos_token_id: nil
      )

    generate =
      Bumblebee.Text.Generation.build_generate(bundle.model, bundle.spec, generation_config)

    outputs = generate.(bundle.params, Map.put(inputs, "seed", Nx.tensor([0])))
    Nx.to_flat_list(outputs.token_ids)
  end
end
