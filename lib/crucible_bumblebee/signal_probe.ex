defmodule CrucibleBumblebee.SignalProbe do
  @moduledoc """
  Native signal probe over real Bumblebee model outputs.
  """

  alias CrucibleBumblebee.{
    Artifacts,
    GenerationTrace,
    Live,
    ModelLoader,
    ModelLoader.Options,
    TraceWriter
  }

  @prompt "Hi"
  @blocked_by_bumblebee :blocked_by_bumblebee_api
  @unsupported_surface :unsupported_by_surface

  def run_model(model, opts \\ []) when is_map(model) do
    root = Keyword.get(opts, :artifact_root)
    backend = Keyword.get(opts, :backend, :binary)
    prompt = Keyword.get(opts, :prompt, @prompt)
    name = "#{model.rung}_#{Artifacts.safe_name(model.model_id)}_#{backend}_signal_probe"
    trace_id = "tr_#{System.unique_integer([:positive])}"
    run_id = "run_#{System.unique_integer([:positive])}"
    trace_path = Artifacts.trace_path(name, root: root)
    started = System.monotonic_time(:millisecond)

    Artifacts.ensure_layout!(root: root)
    TraceWriter.reset!(trace_path)

    TraceWriter.write!(trace_path, :trace_start,
      trace_id: trace_id,
      run_id: run_id,
      provider_kind: :elixir_bumblebee,
      model_id: model.model_id,
      model_family: model.family,
      backend: backend
    )

    try do
      do_run_model(model, backend, root, prompt, trace_path, trace_id, run_id, started)
    rescue
      error ->
        row =
          row(model, backend, :probe, :failed_with_exception, Exception.message(error),
            trace_path: trace_path
          )

        write_row!(row, root)
        %{ok: false, rows: [row], trace_path: trace_path}
    end
  end

  defp do_run_model(model, backend, root, prompt, trace_path, trace_id, run_id, started) do
    bundle =
      ModelLoader.load!(
        Options.new(
          model_id: model.model_id,
          tokenizer_id: Map.get(model, :tokenizer_id, model.model_id),
          backend: backend,
          architecture: model.architecture,
          module: Map.get(model, :module),
          artifact_root: root
        )
      )

    tokenizer = Bumblebee.configure(bundle.tokenizer, return_token_type_ids: false)
    inputs = Bumblebee.apply_tokenizer(tokenizer, prompt)
    outputs = Axon.predict(bundle.model, bundle.params, inputs)

    context = %{
      model: model,
      bundle: bundle,
      backend: backend,
      root: root,
      trace_path: trace_path,
      trace_id: trace_id,
      run_id: run_id,
      inputs: inputs,
      outputs: outputs,
      prompt: prompt
    }

    rows =
      []
      |> capture_input("input_ids", :input_ids, context)
      |> capture_input("attention_mask", :attention_mask, context)
      |> capture_final_logits(context)
      |> classify_generation(context)
      |> classify_hidden_states(context)
      |> classify_attentions(context)
      |> classify_blocked(
        :intermediate_logits,
        @blocked_by_bumblebee,
        :hidden_states_unavailable,
        context
      )
      |> classify_blocked(:residual_stream, @unsupported_surface, @unsupported_surface, context)
      |> classify_blocked(:mlp_activation, @unsupported_surface, @unsupported_surface, context)
      |> classify_blocked(
        :router_logits,
        :unsupported_by_model_family,
        :unsupported_by_model_family,
        context
      )
      |> classify_blocked(
        :moe_expert_weights,
        :unsupported_by_model_family,
        :unsupported_by_model_family,
        context
      )
      |> capture_backend_event(started, context)
      |> Enum.reverse()

    Enum.each(rows, &write_row!(&1, root))

    TraceWriter.write!(trace_path, :trace_end,
      trace_id: trace_id,
      status: :ok,
      duration_ms: elapsed_ms(started)
    )

    %{ok: Enum.any?(rows, &(&1.status == "captured")), rows: rows, trace_path: trace_path}
  end

  defp capture_input(rows, key, signal, context) do
    case Map.fetch(context.inputs, key) do
      {:ok, %Nx.Tensor{} = tensor} ->
        write_signal!(context, tensor, signal,
          signal_id: "sig_#{signal}",
          node_name: key,
          capture_method: :tokenizer_output,
          entropy?: false,
          top_k: 0
        )

        [
          row(context.model, context.backend, signal, :captured, nil,
            trace_path: context.trace_path
          )
          | rows
        ]

      _other ->
        [
          row(
            context.model,
            context.backend,
            signal,
            :blocked_by_bumblebee_api,
            :missing_tokenizer_output,
            trace_path: context.trace_path
          )
          | rows
        ]
    end
  end

  defp capture_final_logits(rows, context) do
    logits = context.outputs |> Live.fetch_logits!() |> Live.last_token_logits()

    signal =
      write_signal!(context, logits, :final_logits,
        signal_id: "sig_final_logits",
        node_name: "final_logits",
        capture_method: :axon_predict,
        entropy?: true,
        top_k: 10
      )

    top_k? = signal.tensor_summary.top_k not in [nil, []]
    entropy? = not is_nil(signal.tensor_summary.entropy)

    rows = [
      row(context.model, context.backend, :final_logits, :captured, nil,
        trace_path: context.trace_path
      )
      | rows
    ]

    rows = [
      row(context.model, context.backend, :top_k_summary, status(top_k?), reason(top_k?),
        trace_path: context.trace_path
      )
      | rows
    ]

    [
      row(context.model, context.backend, :entropy_margin, status(entropy?), reason(entropy?),
        trace_path: context.trace_path
      )
      | rows
    ]
  end

  defp classify_generation(rows, %{model: %{family: family}} = context)
       when family in [:gpt2, :qwen3] do
    case GenerationTrace.run(context.bundle, context.prompt, max_new_tokens: 1) do
      {:ok, %{steps: [step | _rest]}} ->
        write_signal!(context, step.logits, :generation_step_logits,
          signal_id: "sig_generation_step_logits_probe_#{step.step_index}",
          node_name: "generation_step_logits",
          token_index: step.step_index,
          capture_method: :bumblebee_generation_trace,
          metadata: %{cache_metadata: step.cache_metadata},
          entropy?: true,
          top_k: 10
        )

        [
          row(context.model, context.backend, :kv_cache_metadata, :captured, nil,
            trace_path: context.trace_path
          ),
          row(context.model, context.backend, :generation_step_logits, :captured, nil,
            trace_path: context.trace_path
          ),
          row(context.model, context.backend, :generation_token, :captured, nil,
            trace_path: context.trace_path
          )
          | rows
        ]

      {:error, reason} ->
        [
          row(
            context.model,
            context.backend,
            :kv_cache_metadata,
            :failed_with_exception,
            inspect(reason),
            trace_path: context.trace_path
          ),
          row(
            context.model,
            context.backend,
            :generation_step_logits,
            :failed_with_exception,
            inspect(reason),
            trace_path: context.trace_path
          )
          | rows
        ]
    end
  end

  defp classify_generation(rows, context) do
    [
      row(
        context.model,
        context.backend,
        :kv_cache_metadata,
        :unsupported_by_model_family,
        :non_causal_generation,
        trace_path: context.trace_path
      ),
      row(
        context.model,
        context.backend,
        :generation_step_logits,
        :unsupported_by_model_family,
        :non_causal_generation,
        trace_path: context.trace_path
      ),
      row(
        context.model,
        context.backend,
        :generation_token,
        :unsupported_by_model_family,
        :non_causal_generation,
        trace_path: context.trace_path
      )
      | rows
    ]
  end

  defp classify_hidden_states(rows, context) do
    classify_output_collection(rows, context, :hidden_states, :hidden_state)
  end

  defp classify_attentions(rows, context) do
    classify_output_collection(rows, context, :attentions, :attention_weights)
  end

  defp classify_output_collection(rows, context, output_key, signal) do
    value = Map.get(context.outputs, output_key)

    if tensor_collection?(value) do
      [
        row(context.model, context.backend, signal, :captured, nil,
          trace_path: context.trace_path
        )
        | rows
      ]
    else
      [
        row(context.model, context.backend, signal, @blocked_by_bumblebee, :axon_none,
          trace_path: context.trace_path
        )
        | rows
      ]
    end
  end

  defp classify_blocked(rows, signal, status, reason, context) do
    [
      row(context.model, context.backend, signal, status, reason, trace_path: context.trace_path)
      | rows
    ]
  end

  defp capture_backend_event(rows, started, context) do
    TraceWriter.write!(context.trace_path, :backend_event,
      trace_id: context.trace_id,
      backend: context.backend,
      duration_ms: elapsed_ms(started)
    )

    [
      row(context.model, context.backend, :backend_event, :captured, nil,
        trace_path: context.trace_path
      )
      | rows
    ]
  end

  defp write_signal!(context, tensor, signal, attrs) do
    signal_record =
      TraceWriter.signal_from_tensor(
        tensor,
        Map.merge(Map.new(attrs), %{
          trace_id: context.trace_id,
          run_id: context.run_id,
          signal_type: signal,
          model_id: context.bundle.model_id,
          model_family: context.bundle.model_family,
          model_revision: context.bundle.revision,
          backend: context.bundle.backend,
          capability_status: :captured
        })
      )

    TraceWriter.write!(context.trace_path, :signal_record,
      trace_id: context.trace_id,
      signal: signal_record
    )

    signal_record
  end

  defp row(model, backend, signal, status, reason, attrs) do
    %{
      rung: model.rung,
      model_id: model.model_id,
      family: model.family,
      backend: backend,
      signal: signal,
      status: to_string(status),
      reason: if(is_nil(reason), do: nil, else: inspect(reason)),
      trace_event_present: status == :captured,
      policy_uses_it:
        status == :captured and
          signal in [
            :final_logits,
            :generation_step_logits,
            :kv_cache_metadata,
            :top_k_summary,
            :entropy_margin
          ],
      trace: Keyword.get(attrs, :trace_path)
    }
  end

  defp write_row!(row, root) do
    Artifacts.append_jsonl!(:signal_matrix, "signal_probe.jsonl", row, root: root)
    Artifacts.append_jsonl!(:signal_matrix, "signal_ladder.jsonl", row, root: root)
  end

  defp tensor_collection?(%Nx.Tensor{}), do: true

  defp tensor_collection?(value) when is_list(value) or is_tuple(value) do
    value
    |> tuple_or_list()
    |> Enum.any?(&match?(%Nx.Tensor{}, &1))
  end

  defp tensor_collection?(_value), do: false

  defp tuple_or_list(value) when is_tuple(value), do: Tuple.to_list(value)
  defp tuple_or_list(value), do: value

  defp status(true), do: :captured
  defp status(false), do: :blocked_by_bumblebee_api

  defp reason(true), do: nil
  defp reason(false), do: :summary_unavailable

  defp elapsed_ms(start_ms), do: System.monotonic_time(:millisecond) - start_ms
end
