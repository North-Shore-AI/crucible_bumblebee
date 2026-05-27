defmodule CrucibleBumblebee.ManualGeneration do
  @moduledoc """
  Manual autoregressive generation over repeated Bumblebee forward calls.

  Bumblebee's high-level generation serving returns decoded text, but it does
  not expose per-step logits through the current runner. This module provides
  the V5 fallback path: run the model once per generated token, greedily append
  the selected token, and return bounded summaries for each step.
  """

  alias CrucibleBumblebee.{Live, ModelBundle}

  def run(%ModelBundle{} = bundle, prompt, opts \\ []) when is_binary(prompt) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 1)

    tokenizer =
      Bumblebee.configure(bundle.tokenizer,
        return_token_type_ids: false
      )

    initial_inputs = Bumblebee.apply_tokenizer(tokenizer, prompt)

    {inputs, steps} =
      Enum.reduce(1..max_new_tokens, {initial_inputs, []}, fn step_index, {inputs, steps} ->
        outputs = Axon.predict(bundle.model, bundle.params, inputs)
        logits = outputs |> Live.fetch_logits!() |> Live.last_token_logits()
        token_id = greedy_token_id(logits)
        signal_summary = Crucible.TensorSummary.compute(logits, entropy: true, top_k: 10)

        step = %{
          step_index: step_index,
          token_id: token_id,
          token_text: decode_token(bundle.tokenizer, token_id),
          logits: logits,
          tensor_summary: signal_summary,
          entropy: signal_summary.entropy,
          margin: margin(signal_summary.top_k),
          top_k: signal_summary.top_k
        }

        {append_token(inputs, token_id), [step | steps]}
      end)

    steps = Enum.reverse(steps)

    {:ok,
     %{
       steps: steps,
       generated_token_ids: Enum.map(steps, & &1.token_id),
       decoded_text: decode_tokens(bundle.tokenizer, Enum.map(steps, & &1.token_id)),
       final_inputs: inputs
     }}
  rescue
    error -> {:error, {:manual_generation_exception, Exception.message(error)}}
  end

  def append_token(inputs, token_id) when is_map(inputs) and is_integer(token_id) do
    input_ids = Map.fetch!(inputs, "input_ids")
    token = Nx.tensor([[token_id]], type: Nx.type(input_ids))

    inputs
    |> Map.put("input_ids", Nx.concatenate([input_ids, token], axis: 1))
    |> append_attention_mask()
  end

  def greedy_token_id(%Nx.Tensor{} = logits) do
    axis = tuple_size(Nx.shape(logits)) - 1

    logits
    |> Nx.argmax(axis: axis)
    |> Nx.reshape({})
    |> Nx.to_number()
  end

  def public_step(step) when is_map(step) do
    step
    |> Map.drop([:logits])
    |> Map.update(:tensor_summary, nil, &Map.from_struct/1)
  end

  defp append_attention_mask(%{"attention_mask" => attention_mask} = inputs) do
    one = Nx.tensor([[1]], type: Nx.type(attention_mask))
    Map.put(inputs, "attention_mask", Nx.concatenate([attention_mask, one], axis: 1))
  end

  defp append_attention_mask(inputs), do: inputs

  defp decode_token(tokenizer, token_id) do
    Bumblebee.Tokenizer.decode(tokenizer, [token_id])
  rescue
    _error -> nil
  end

  defp decode_tokens(_tokenizer, []), do: ""

  defp decode_tokens(tokenizer, token_ids) do
    Bumblebee.Tokenizer.decode(tokenizer, token_ids)
  rescue
    _error -> nil
  end

  defp margin([top1, top2 | _rest]) do
    Map.fetch!(top1, :logit) - Map.fetch!(top2, :logit)
  end

  defp margin(_top_k), do: nil
end
