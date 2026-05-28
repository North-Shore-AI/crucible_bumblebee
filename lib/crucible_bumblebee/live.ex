defmodule CrucibleBumblebee.Live do
  @moduledoc """
  Standalone V4 native tiny GPT-2 live gates.
  """

  alias CrucibleBumblebee.{ManualGeneration, ModelLoader, Preflight, TraceWriter}
  alias CrucibleBumblebee.ModelLoader.Options
  alias CrucibleTap.TapPlan

  @default_prompt "Hi"

  def forward(opts \\ []) do
    name = Keyword.get(opts, :name, "model_forward_live")
    options = Options.from_env(opts)
    prompt = Keyword.get(opts, :prompt, options.prompt || @default_prompt)
    {:ok, selected_backend} = CrucibleBumblebee.Backend.prefer(options.backend)
    trace_id = "tr_#{System.unique_integer([:positive])}"
    run_id = "run_#{System.unique_integer([:positive])}"
    trace_path = TraceWriter.output_path(name, "trace.jsonl")
    report_path = TraceWriter.output_path(name, "capability_report.json")
    TraceWriter.reset!(trace_path)

    start = System.monotonic_time(:millisecond)

    TraceWriter.write!(trace_path, :trace_start,
      trace_id: trace_id,
      run_id: run_id,
      provider_kind: :elixir_bumblebee,
      model_id: options.model_id,
      model_family: ModelLoader.infer_model_family(options.model_id),
      backend: selected_backend
    )

    tap_plan = forward_tap_plan()
    static_report = static_capability_report(options.model_id, selected_backend)
    TraceWriter.write_capability_report!(report_path, static_report)

    TraceWriter.write!(trace_path, :provider_capability_report,
      trace_id: trace_id,
      capability_report: static_report,
      capability_report_digest: Crucible.CanonicalJSON.digest(static_report)
    )

    TraceWriter.write!(trace_path, :model_load_start,
      trace_id: trace_id,
      model_id: options.model_id
    )

    load_started = System.monotonic_time(:millisecond)
    bundle = ModelLoader.load!(%{options | backend: selected_backend})

    TraceWriter.write!(trace_path, :model_load_end,
      trace_id: trace_id,
      model_loaded: true,
      duration_ms: elapsed_ms(load_started)
    )

    TraceWriter.write!(trace_path, :tokenizer_load_end,
      trace_id: trace_id,
      tokenizer_loaded: true,
      duration_ms: 0
    )

    TraceWriter.write!(trace_path, :tap_compile_start,
      trace_id: trace_id,
      tap_plan_digest: Crucible.CanonicalJSON.digest(tap_plan)
    )

    compile_started = System.monotonic_time(:millisecond)
    preflight = Preflight.run!(bundle, tap_plan)
    report = preflight.capability_report
    TraceWriter.write_capability_report!(report_path, report)

    TraceWriter.write!(trace_path, :tap_compile_end,
      trace_id: trace_id,
      compiled_taps: report.supported,
      optional_dropped: report.optional_dropped,
      duration_ms: elapsed_ms(compile_started)
    )

    TraceWriter.write!(trace_path, :forward_start,
      trace_id: trace_id,
      prompt_digest: CrucibleSignalTrace.Digest.prefixed_text(prompt)
    )

    forward_started = System.monotonic_time(:millisecond)
    logits = run_logits(bundle, prompt)

    signal =
      TraceWriter.signal_from_logits(logits, %{
        signal_id: "sig_final_logits",
        trace_id: trace_id,
        run_id: run_id,
        model_id: bundle.model_id,
        model_family: bundle.model_family,
        backend: bundle.backend
      })

    TraceWriter.write!(trace_path, :signal_record, trace_id: trace_id, signal: signal)

    TraceWriter.write!(trace_path, :forward_end,
      trace_id: trace_id,
      forward_time_ms: elapsed_ms(forward_started)
    )

    TraceWriter.write!(trace_path, :trace_end,
      trace_id: trace_id,
      status: :ok,
      duration_ms: elapsed_ms(start)
    )

    %{
      ok: true,
      example: "model_forward_live",
      provider_kind: :elixir_bumblebee,
      model_id: bundle.model_id,
      tokenizer_loaded?: true,
      model_loaded?: true,
      forward_pass_ran?: true,
      backend: bundle.backend,
      trace_path: trace_path,
      capability_report_path: report_path,
      available_signals: [:final_logits],
      unavailable_signals: [:token_embeddings, :hidden_state, :attention_weights],
      signal_count: 1
    }
  end

  def generation(opts \\ []) do
    name = Keyword.get(opts, :name, "model_generation_live")
    options = Options.from_env(opts)
    prompt = Keyword.get(opts, :prompt, options.prompt || @default_prompt)
    max_new_tokens = Keyword.get(opts, :max_new_tokens) || options.max_new_tokens || 1
    attempt_high_level? = Keyword.get(opts, :attempt_high_level_generation?, false)
    generation_strategy = Keyword.get(opts, :generation_strategy, :greedy)
    stop_token_ids = Keyword.get(opts, :stop_token_ids, [])
    {:ok, selected_backend} = CrucibleBumblebee.Backend.prefer(options.backend)
    trace_id = "tr_#{System.unique_integer([:positive])}"
    run_id = "run_#{System.unique_integer([:positive])}"
    trace_path = TraceWriter.output_path(name, "trace.jsonl")
    report_path = TraceWriter.output_path(name, "capability_report.json")
    TraceWriter.reset!(trace_path)

    TraceWriter.write!(trace_path, :trace_start,
      trace_id: trace_id,
      run_id: run_id,
      provider_kind: :elixir_bumblebee,
      model_id: options.model_id,
      model_family: ModelLoader.infer_model_family(options.model_id),
      backend: selected_backend
    )

    report =
      static_capability_report(options.model_id, selected_backend,
        supported: [
          :final_logits,
          :generation_token,
          :generation_step_logits,
          :decode_entropy,
          :decode_margin
        ],
        unsupported: [
          %Crucible.UnsupportedCapability{
            capability: :hidden_state,
            reason: :unsupported_by_surface,
            required?: false
          },
          %Crucible.UnsupportedCapability{
            capability: :attention_weights,
            reason: :unsupported_by_surface,
            required?: false
          },
          %Crucible.UnsupportedCapability{
            capability: :kv_cache_metadata,
            reason: :blocked_by_generation_pipeline,
            required?: false
          }
        ],
        optional_dropped: [:hidden_state, :attention_weights, :kv_cache_metadata]
      )

    TraceWriter.write_capability_report!(report_path, report)

    TraceWriter.write!(trace_path, :provider_capability_report,
      trace_id: trace_id,
      capability_report: report,
      capability_report_digest: Crucible.CanonicalJSON.digest(report)
    )

    TraceWriter.write!(trace_path, :model_load_start,
      trace_id: trace_id,
      model_id: options.model_id
    )

    load_started = System.monotonic_time(:millisecond)
    bundle = ModelLoader.load!(%{options | backend: selected_backend})

    TraceWriter.write!(trace_path, :model_load_end,
      trace_id: trace_id,
      model_loaded: true,
      duration_ms: elapsed_ms(load_started)
    )

    TraceWriter.write!(trace_path, :generation_start,
      trace_id: trace_id,
      prompt_digest: CrucibleSignalTrace.Digest.prefixed_text(prompt),
      max_new_tokens: max_new_tokens,
      decode_mode: generation_strategy
    )

    high_level_text = if attempt_high_level?, do: maybe_generate_text(bundle, prompt)

    case ManualGeneration.run(bundle, prompt,
           max_new_tokens: max_new_tokens,
           strategy: generation_strategy,
           seed: options.seed,
           stop_token_ids: stop_token_ids
         ) do
      {:ok, manual} ->
        Enum.each(manual.steps, fn step ->
          signal =
            TraceWriter.signal_from_tensor(step.logits, %{
              signal_id: "sig_generation_step_logits_#{step.step_index}",
              trace_id: trace_id,
              run_id: run_id,
              signal_type: :generation_step_logits,
              model_id: bundle.model_id,
              model_family: bundle.model_family,
              model_revision: bundle.revision,
              backend: bundle.backend,
              token_index: step.step_index,
              node_name: "generation_step_logits",
              capture_method: :manual_autoregressive_loop,
              capability_status: :captured
            })

          TraceWriter.write!(trace_path, :signal_record, trace_id: trace_id, signal: signal)

          TraceWriter.write!(trace_path, :generation_step,
            trace_id: trace_id,
            step_index: step.step_index,
            generated_token_id: step.token_id,
            generated_token_text: step.token_text,
            logits_signal_id: signal.signal_id,
            entropy: step.entropy,
            margin: step.margin,
            top_k: step.top_k
          )
        end)

        TraceWriter.write!(trace_path, :generation_end,
          trace_id: trace_id,
          status: :ok,
          high_level_generation_text_available?: not is_nil(high_level_text),
          generated_token_ids: manual.generated_token_ids,
          generated_text: manual.decoded_text
        )

        TraceWriter.write!(trace_path, :trace_end, trace_id: trace_id, status: :ok)

        %{
          ok: true,
          generation_supported?: true,
          generation_success_level: :generation_step_logits,
          generated_token_ids: manual.generated_token_ids,
          generated_text: manual.decoded_text,
          step_count: length(manual.steps),
          capability_report_path: report_path,
          trace_path: trace_path
        }

      {:error, reason} ->
        TraceWriter.write!(trace_path, :capability_blocker,
          trace_id: trace_id,
          capability: :generation_step_logits,
          reason: reason
        )

        TraceWriter.write!(trace_path, :generation_end,
          trace_id: trace_id,
          status: :failed,
          reason: reason
        )

        TraceWriter.write!(trace_path, :trace_end, trace_id: trace_id, status: :failed)

        %{
          ok: false,
          generation_supported?: false,
          reason: reason,
          capability_report_path: report_path,
          trace_path: trace_path
        }
    end
  end

  def forward_tap_plan do
    TapPlan.new!(
      [
        [id: "final_logits", signal_type: :final_logits, layers: [:final], required?: true],
        [id: "hidden_state", signal_type: :hidden_state, layers: [:last], required?: false],
        [
          id: "attention_weights",
          signal_type: :attention_weights,
          layers: [:last],
          required?: false
        ]
      ],
      plan_id: "v4-native-tiny-gpt2-forward"
    )
  end

  def static_capability_report(model_id, backend, attrs \\ []) do
    supported = Keyword.get(attrs, :supported, [:final_logits])

    unsupported =
      Keyword.get(attrs, :unsupported, [
        %Crucible.UnsupportedCapability{
          capability: :hidden_state,
          reason: :unsupported_by_surface,
          required?: false
        },
        %Crucible.UnsupportedCapability{
          capability: :attention_weights,
          reason: :unsupported_by_surface,
          required?: false
        }
      ])

    optional_dropped =
      Keyword.get(attrs, :optional_dropped, Enum.map(unsupported, & &1.capability))

    %Crucible.CapabilityReport{
      provider_kind: :elixir_bumblebee,
      model_id: model_id,
      model_family: ModelLoader.infer_model_family(model_id),
      backend: backend,
      supported: supported,
      unsupported: unsupported,
      failed: [],
      degraded:
        Enum.map(unsupported, fn unsupported ->
          %Crucible.DegradedCapability{
            capability: unsupported.capability,
            reason: unsupported.reason,
            required?: unsupported.required?
          }
        end),
      resource_budget: Preflight.resource_budget(),
      required_missing: [],
      optional_dropped: optional_dropped
    }
  end

  def run_logits(bundle, prompt) do
    tokenizer =
      Bumblebee.configure(bundle.tokenizer,
        return_token_type_ids: false
      )

    inputs = Bumblebee.apply_tokenizer(tokenizer, prompt)
    outputs = Axon.predict(bundle.model, bundle.params, inputs)

    outputs
    |> fetch_logits!()
    |> last_token_logits()
  end

  def fetch_logits!(%{logits: logits}), do: logits
  def fetch_logits!(%{"logits" => logits}), do: logits

  def fetch_logits!(outputs),
    do: raise("model output did not contain logits: #{inspect(Map.keys(outputs))}")

  def last_token_logits(%Nx.Tensor{} = logits) do
    case Nx.shape(logits) do
      {_batch, sequence_length, _vocab} ->
        Nx.slice_along_axis(logits, sequence_length - 1, 1, axis: 1)

      _shape ->
        logits
    end
  end

  defp maybe_generate_text(%{generation_config: nil}, _prompt), do: nil

  defp maybe_generate_text(bundle, prompt) do
    generation_config = Bumblebee.configure(bundle.generation_config, max_new_tokens: 2)

    serving =
      Bumblebee.Text.generation(
        %{model: bundle.model, params: bundle.params, spec: bundle.spec},
        bundle.tokenizer,
        generation_config,
        compile: [batch_size: 1, sequence_length: 16]
      )

    Nx.Serving.run(serving, prompt)
  rescue
    _error -> nil
  end

  defp elapsed_ms(start_ms), do: System.monotonic_time(:millisecond) - start_ms
end
