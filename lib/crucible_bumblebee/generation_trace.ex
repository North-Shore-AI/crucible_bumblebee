defmodule CrucibleBumblebee.GenerationTrace do
  @moduledoc """
  KV-cache-backed autoregressive generation traces.

  This module uses the fork-backed `Bumblebee.Text.Generation.build_generate/4`
  trace output. The generation itself is Bumblebee's normal cached decode path;
  this adapter only converts the returned token IDs, processed step logits, and
  cache offsets into bounded Crucible trace records.
  """

  alias CrucibleBumblebee.ModelBundle

  @success_level :kv_cache_generation_trace

  def run(%ModelBundle{} = bundle, prompt, opts \\ []) when is_binary(prompt) do
    try do
      tokenizer = Bumblebee.configure(bundle.tokenizer, return_token_type_ids: false)
      inputs = Bumblebee.apply_tokenizer(tokenizer, prompt)
      run_inputs(bundle, inputs, opts)
    rescue
      error -> {:error, {:generation_trace_exception, Exception.message(error)}}
    end
  end

  def run_inputs(%ModelBundle{} = bundle, inputs, opts \\ []) when is_map(inputs) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 1)
    top_k = Keyword.get(opts, :top_k, 10)

    with :ok <- validate_max_new_tokens(max_new_tokens),
         :ok <- validate_strategy(opts),
         :ok <- validate_single_batch_inputs(inputs) do
      prompt_length = prompt_length(inputs)

      if max_new_tokens == 0 do
        {:ok,
         %{
           steps: [],
           generated_token_ids: [],
           decoded_text: "",
           generation_success_level: @success_level,
           optional_internals: :not_requested,
           trace_metadata: %{
             prompt_length: prompt_length,
             requested_max_new_tokens: 0,
             emitted_steps: 0,
             cache_offset_source: :bumblebee_generation_trace
           }
         }}
      else
        generation_config = generation_config(bundle, opts)

        generate =
          Bumblebee.Text.Generation.build_generate(
            bundle.model,
            bundle.spec,
            generation_config,
            trace: true
          )

        outputs = generate.(bundle.params, put_seed(inputs, Keyword.get(opts, :seed, 0)))
        steps = build_steps(outputs, bundle, prompt_length, max_new_tokens, top_k)
        generated_token_ids = Enum.map(steps, & &1.token_id)

        {:ok,
         %{
           steps: steps,
           generated_token_ids: generated_token_ids,
           decoded_text: decode_tokens(bundle.tokenizer, generated_token_ids),
           generation_success_level: @success_level,
           optional_internals: :not_requested,
           trace_metadata: %{
             prompt_length: prompt_length,
             requested_max_new_tokens: max_new_tokens,
             emitted_steps: length(steps),
             cache_offset_source: :bumblebee_generation_trace
           }
         }}
      end
    end
  rescue
    error -> {:error, {:generation_trace_exception, Exception.message(error)}}
  end

  def public_step(step) when is_map(step) do
    step
    |> Map.drop([:logits])
    |> Map.update(:tensor_summary, nil, &Map.from_struct/1)
  end

  defp build_steps(outputs, bundle, prompt_length, max_new_tokens, top_k) do
    generated_count = generated_count(outputs.length, max_new_tokens)
    token_ids = output_token_ids(outputs.token_ids, generated_count)

    token_ids
    |> Enum.with_index()
    |> Enum.map(fn {token_id, index} ->
      step_index = index + 1
      logits = trace_logits(outputs.trace_logits, index)
      summary = Crucible.TensorSummary.compute(logits, entropy: true, top_k: top_k)
      cache_offset = trace_cache_offset(outputs.trace_cache_offsets, index)

      %{
        step_index: step_index,
        token_index: prompt_length + index,
        token_id: token_id,
        token_text: decode_token(bundle.tokenizer, token_id),
        logits: logits,
        tensor_summary: summary,
        entropy: summary.entropy,
        margin: margin(summary.top_k),
        top_k: summary.top_k,
        cache_offset: cache_offset,
        cache_metadata: %{
          type: :decoder_kv_cache,
          offset: cache_offset,
          prompt_length: prompt_length,
          generated_length: step_index,
          max_length: prompt_length + max_new_tokens,
          batch_index: 0,
          source: :bumblebee_generation_trace
        }
      }
    end)
  end

  defp generation_config(%ModelBundle{} = bundle, opts) do
    base = bundle.generation_config || Bumblebee.Text.GenerationConfig
    Bumblebee.configure(base, generation_config_options(bundle, opts))
  end

  defp generation_config_options(%ModelBundle{} = bundle, opts) do
    options = [
      max_new_tokens: Keyword.get(opts, :max_new_tokens, 1),
      strategy: %{type: :greedy_search}
    ]

    options =
      if is_nil(bundle.generation_config) do
        Keyword.put(options, :pad_token_id, Keyword.get(opts, :pad_token_id, 0))
      else
        put_if_present(options, opts, :pad_token_id)
      end

    options
    |> put_if_present(opts, :eos_token_id)
    |> put_stop_token_ids(opts)
  end

  defp put_if_present(options, opts, key) do
    if Keyword.has_key?(opts, key),
      do: Keyword.put(options, key, Keyword.fetch!(opts, key)),
      else: options
  end

  defp put_stop_token_ids(options, opts) do
    stop_token_ids = Keyword.get(opts, :stop_token_ids, [])

    cond do
      stop_token_ids == [] -> options
      Keyword.has_key?(options, :eos_token_id) -> options
      true -> Keyword.put(options, :eos_token_id, stop_token_ids)
    end
  end

  defp put_seed(inputs, seed) do
    batch_size = inputs |> Map.fetch!("input_ids") |> Nx.axis_size(0)
    Map.put_new_lazy(inputs, "seed", fn -> Nx.broadcast(seed, {batch_size}) end)
  end

  defp generated_count(length, max_new_tokens) do
    length
    |> Nx.to_flat_list()
    |> List.first()
    |> min(max_new_tokens)
    |> max(0)
  end

  defp output_token_ids(_token_ids, 0), do: []

  defp output_token_ids(token_ids, count) do
    token_ids
    |> Nx.slice([0, 0], [1, count])
    |> Nx.reshape({count})
    |> Nx.to_flat_list()
  end

  defp trace_logits(trace_logits, step_index) do
    vocab_size = Nx.axis_size(trace_logits, 2)

    trace_logits
    |> Nx.slice([0, step_index, 0], [1, 1, vocab_size])
    |> Nx.reshape({vocab_size})
  end

  defp trace_cache_offset(trace_cache_offsets, step_index) do
    trace_cache_offsets
    |> Nx.slice([0, step_index], [1, 1])
    |> Nx.reshape({})
    |> Nx.to_number()
  end

  defp decode_token(nil, _token_id), do: nil

  defp decode_token(tokenizer, token_id) do
    Bumblebee.Tokenizer.decode(tokenizer, [token_id])
  rescue
    _error -> nil
  end

  defp decode_tokens(_tokenizer, []), do: ""
  defp decode_tokens(nil, _token_ids), do: nil

  defp decode_tokens(tokenizer, token_ids) do
    Bumblebee.Tokenizer.decode(tokenizer, token_ids)
  rescue
    _error -> nil
  end

  defp margin([top1, top2 | _rest]) do
    Map.fetch!(top1, :logit) - Map.fetch!(top2, :logit)
  end

  defp margin(_top_k), do: nil

  defp prompt_length(inputs), do: inputs |> Map.fetch!("input_ids") |> Nx.axis_size(1)

  defp validate_max_new_tokens(value) when is_integer(value) and value >= 0, do: :ok

  defp validate_max_new_tokens(value),
    do: {:error, {:invalid_max_new_tokens, value}}

  defp validate_strategy(opts) do
    case Keyword.get(opts, :strategy, :greedy) do
      :greedy -> :ok
      :greedy_search -> :ok
      %{type: :greedy_search} -> :ok
      other -> {:error, {:unsupported_generation_trace_strategy, other}}
    end
  end

  defp validate_single_batch_inputs(inputs) do
    case Nx.shape(Map.fetch!(inputs, "input_ids")) do
      {1, _sequence_length} ->
        :ok

      {batch_size, _sequence_length} ->
        {:error, {:batch_generation_trace_not_yet_supported, batch_size}}

      shape ->
        {:error, {:invalid_input_ids_shape, shape}}
    end
  end
end
