defmodule CrucibleBumblebee.LiveMatrix do
  @moduledoc """
  V5 live model/backend ladder runner.
  """

  alias CrucibleBumblebee.{Artifacts, Backend, Live}

  @backend_ladder [:binary, :exla_cpu, :torchx_cpu, :exla_cuda]

  @model_ladder [
    %{
      rung: "M0",
      model_id: "hf-internal-testing/tiny-random-gpt2",
      family: :gpt2,
      architecture: :for_causal_language_modeling,
      required?: true
    },
    %{
      rung: "M1",
      model_id: "gpt2",
      family: :gpt2,
      architecture: :for_causal_language_modeling,
      required?: true
    },
    %{
      rung: "M2",
      model_id: "distilgpt2",
      family: :gpt2,
      architecture: :for_causal_language_modeling,
      required?: true
    },
    %{
      rung: "M3",
      model_id: "hf-internal-testing/tiny-random-OPTForCausalLM",
      family: :opt,
      architecture: :for_causal_language_modeling,
      expected_blocker: :unsupported_by_bumblebee_module,
      generation_expected_blocker: :unsupported_by_bumblebee_module,
      required?: true
    },
    %{
      rung: "M4",
      model_id: "hf-internal-testing/tiny-random-BertModel",
      family: :bert,
      architecture: :base,
      module: Bumblebee.Text.Bert,
      expected_blocker: :unsupported_live_surface_family,
      generation_expected_blocker: :non_causal_generation,
      required?: true
    },
    %{
      rung: "M5",
      model_id: "hf-internal-testing/tiny-random-distilbert",
      family: :distilbert,
      architecture: :for_sequence_classification,
      module: Bumblebee.Text.Distilbert,
      generation_expected_blocker: :non_causal_generation,
      required?: true
    },
    %{
      rung: "M6",
      model_id: "trl-internal-testing/tiny-Qwen3ForCausalLM",
      family: :qwen3,
      architecture: :for_causal_language_modeling,
      module: Bumblebee.Text.Qwen3,
      required?: true
    }
  ]

  def model_ladder, do: @model_ladder
  def backend_ladder, do: @backend_ladder

  def run_model_ladder(opts \\ []) do
    root = Keyword.get(opts, :artifact_root)
    backend = Keyword.get(opts, :backend, :binary)
    models = Keyword.get(opts, :models, @model_ladder)
    attempt_fun = Keyword.get(opts, :attempt_fun, &attempt_forward/3)

    Artifacts.ensure_layout!(root: root)

    rows =
      Enum.map(models, fn model ->
        row = attempt_fun.(model, backend, root)
        write_rows!(row, root)
        row
      end)

    %{
      ok: Enum.all?(rows, &(&1.result in ["passed", "blocked_expected"])),
      backend: backend,
      rows: rows
    }
  end

  def run_generation_ladder(opts \\ []) do
    root = Keyword.get(opts, :artifact_root)
    backend = Keyword.get(opts, :backend, :binary)
    models = Keyword.get(opts, :models, @model_ladder)
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 1)
    attempt_high_level? = Keyword.get(opts, :attempt_high_level_generation?, false)
    attempt_fun = Keyword.get(opts, :attempt_fun, &attempt_generation/4)

    Artifacts.ensure_layout!(root: root)

    rows =
      Enum.map(models, fn model ->
        row =
          attempt_fun.(
            model,
            backend,
            [max_new_tokens: max_new_tokens, attempt_high_level_generation?: attempt_high_level?],
            root
          )

        write_generation_row!(row, root)
        row
      end)

    %{
      ok: Enum.all?(rows, &(&1.result in ["passed", "blocked_expected"])),
      backend: backend,
      rows: rows
    }
  end

  def run_backend_ladder(opts \\ []) do
    root = Keyword.get(opts, :artifact_root)
    models = Keyword.get(opts, :models, @model_ladder)
    backends = Keyword.get(opts, :backends, @backend_ladder)
    backend_probe_fun = Keyword.get(opts, :backend_probe_fun, &Backend.prefer/1)
    attempt_fun = Keyword.get(opts, :attempt_fun, &attempt_forward/3)

    Artifacts.ensure_layout!(root: root)

    rows =
      for backend <- backends, model <- models do
        row =
          case backend_probe_fun.(backend) do
            {:ok, selected_backend} ->
              Backend.reset()
              attempt_fun.(model, selected_backend, root)

            {:error, reason} ->
              backend_unavailable_row(model, backend, reason)
          end

        write_backend_ladder_row!(row, root)
        row
      end

    %{
      ok: Enum.all?(rows, &(&1.result in ["passed", "blocked_expected", "backend_unavailable"])),
      rows: rows
    }
  end

  def attempt_forward(model, backend, root) do
    name = "#{model.rung}_#{Artifacts.safe_name(model.model_id)}_#{backend}"
    started = System.monotonic_time(:millisecond)

    try do
      do_attempt_forward(model, backend, root, name, started)
    rescue
      error -> failed_row(model, backend, started, error)
    end
  end

  def attempt_generation(model, backend, generation_opts, root) do
    name = "#{model.rung}_#{Artifacts.safe_name(model.model_id)}_#{backend}_generation"
    started = System.monotonic_time(:millisecond)
    max_new_tokens = Keyword.fetch!(generation_opts, :max_new_tokens)

    try do
      with :ok <- generation_supported?(model) do
        result =
          Live.generation(
            name: name,
            model_id: model.model_id,
            tokenizer_id: Map.get(model, :tokenizer_id, model.model_id),
            backend: backend,
            architecture: model.architecture,
            module: Map.get(model, :module),
            artifact_root: root,
            max_new_tokens: max_new_tokens,
            attempt_high_level_generation?:
              Keyword.get(generation_opts, :attempt_high_level_generation?, false)
          )

        %{
          rung: model.rung,
          model_id: model.model_id,
          family: model.family,
          backend: backend,
          generation: result.generation_supported?,
          success_level: result.generation_success_level,
          generated_tokens: Map.get(result, :generated_token_ids, []),
          step_logits: result.generation_success_level == :generation_step_logits,
          step_count: Map.get(result, :step_count, 0),
          trace: result.trace_path,
          duration_ms: elapsed_ms(started),
          result: if(result.ok, do: "passed", else: "failed"),
          error: if(result.ok, do: nil, else: inspect(Map.get(result, :reason)))
        }
      else
        {:error, blocker} -> blocked_generation_row(model, backend, started, blocker)
      end
    rescue
      error -> failed_generation_row(model, backend, started, error)
    end
  end

  defp generation_supported?(%{generation_expected_blocker: blocker}), do: {:error, blocker}
  defp generation_supported?(%{family: family}) when family in [:gpt2, :qwen3], do: :ok
  defp generation_supported?(%{family: family}), do: {:error, {:non_causal_generation, family}}

  defp blocked_generation_row(model, backend, started, blocker) do
    %{
      rung: model.rung,
      model_id: model.model_id,
      family: model.family,
      backend: backend,
      generation: false,
      success_level: :generation_failed,
      generated_tokens: [],
      step_logits: false,
      step_count: 0,
      trace: nil,
      duration_ms: elapsed_ms(started),
      result: "blocked_expected",
      error: inspect(blocker),
      blocker: blocker
    }
  end

  defp do_attempt_forward(model, backend, root, name, started) do
    result =
      Live.forward(
        name: name,
        model_id: model.model_id,
        tokenizer_id: Map.get(model, :tokenizer_id, model.model_id),
        backend: backend,
        architecture: model.architecture,
        module: Map.get(model, :module),
        artifact_root: root
      )

    %{
      rung: model.rung,
      model_id: model.model_id,
      family: model.family,
      backend: backend,
      tokenizer_loaded: result.tokenizer_loaded?,
      model_loaded: result.model_loaded?,
      forward: result.forward_pass_ran?,
      final_logits: :final_logits in result.available_signals,
      generation: false,
      step_logits: false,
      hidden_states: :hidden_state in result.available_signals,
      attention: :attention_weights in result.available_signals,
      trace: result.trace_path,
      policy_replay: false,
      trinity_trace: false,
      trinity_live: false,
      duration_ms: elapsed_ms(started),
      result: "passed",
      error: nil
    }
  end

  defp failed_row(model, backend, started, error) do
    expected? = Map.has_key?(model, :expected_blocker)

    %{
      rung: model.rung,
      model_id: model.model_id,
      family: model.family,
      backend: backend,
      tokenizer_loaded: false,
      model_loaded: false,
      forward: false,
      final_logits: false,
      generation: false,
      step_logits: false,
      hidden_states: false,
      attention: false,
      trace: nil,
      policy_replay: false,
      trinity_trace: false,
      trinity_live: false,
      duration_ms: elapsed_ms(started),
      result: if(expected?, do: "blocked_expected", else: "failed"),
      error: Exception.message(error),
      blocker: Map.get(model, :expected_blocker)
    }
  end

  defp write_rows!(row, root) do
    Artifacts.append_jsonl!(:model_matrix, "model_ladder.jsonl", row, root: root)

    Artifacts.append_jsonl!(
      :backend_matrix,
      "backend_ladder.jsonl",
      Map.take(row, [
        :rung,
        :model_id,
        :family,
        :backend,
        :model_loaded,
        :forward,
        :generation,
        :duration_ms,
        :result,
        :error,
        :blocker
      ]),
      root: root
    )

    Artifacts.append_jsonl!(
      :signal_matrix,
      "signal_ladder.jsonl",
      %{
        model_id: row.model_id,
        backend: row.backend,
        final_logits: row.final_logits,
        hidden_states: row.hidden_states,
        attention: row.attention,
        step_logits: row.step_logits,
        result: row.result
      },
      root: root
    )
  end

  defp failed_generation_row(model, backend, started, error) do
    expected? = Map.has_key?(model, :expected_blocker)

    %{
      rung: model.rung,
      model_id: model.model_id,
      family: model.family,
      backend: backend,
      generation: false,
      success_level: :generation_failed,
      generated_tokens: [],
      step_logits: false,
      step_count: 0,
      trace: nil,
      duration_ms: elapsed_ms(started),
      result: if(expected?, do: "blocked_expected", else: "failed"),
      error: Exception.message(error),
      blocker: Map.get(model, :expected_blocker)
    }
  end

  defp write_generation_row!(row, root) do
    Artifacts.append_jsonl!(:generation_matrix, "generation_ladder.jsonl", row, root: root)
  end

  defp backend_unavailable_row(model, backend, reason) do
    %{
      rung: model.rung,
      model_id: model.model_id,
      family: model.family,
      backend: backend,
      tokenizer_loaded: false,
      model_loaded: false,
      forward: false,
      generation: false,
      final_logits: false,
      step_logits: false,
      hidden_states: false,
      attention: false,
      duration_ms: 0,
      result: "backend_unavailable",
      error: inspect(reason),
      blocker: json_safe(reason)
    }
  end

  defp write_backend_ladder_row!(row, root) do
    Artifacts.append_jsonl!(
      :backend_matrix,
      "backend_ladder.jsonl",
      Map.take(row, [
        :rung,
        :model_id,
        :family,
        :backend,
        :model_loaded,
        :forward,
        :generation,
        :duration_ms,
        :result,
        :error,
        :blocker
      ]),
      root: root
    )
  end

  defp json_safe(value) when is_tuple(value),
    do: value |> Tuple.to_list() |> Enum.map(&json_safe/1)

  defp json_safe(value) when is_list(value), do: Enum.map(value, &json_safe/1)

  defp json_safe(value) when is_map(value),
    do: Map.new(value, fn {key, value} -> {key, json_safe(value)} end)

  defp json_safe(value), do: value

  defp elapsed_ms(start_ms), do: System.monotonic_time(:millisecond) - start_ms
end
