defmodule CrucibleBumblebee.EMLXQwen3 do
  @moduledoc """
  Crucible bridge for the native EMLX Qwen3 trace surface.

  This module intentionally does not depend on `emlx_axon` at compile time. The
  EMLX repo owns execution and lazy MLX tensor behavior; this bridge owns the
  Crucible-facing surface metadata and conversion of EMLX trace maps into
  Crucible cache structures.
  """

  alias CrucibleBumblebee.{ModelSurface, Qwen3Surface}
  alias CrucibleMechInterp.{ActivationCache, ActivationSpec}

  @emlx_repo "https://github.com/North-Shore-AI/emlx.git"
  @emlx_branch "phase-9-qwen3-trace"
  @emlx_ref "4076ec422e38255bc7cda3fb527ea372d010351e"

  @capture_groups [
    :cache_metadata,
    :final_logits,
    :residual_streams,
    :attention_qkv,
    :attention_scores,
    :attention_pattern,
    :attention_z,
    :mlp_activations,
    :norm_telemetry
  ]

  @doc "Pinned EMLX fork state required by this bridge."
  def dependency_pin do
    %{
      repo: @emlx_repo,
      branch: @emlx_branch,
      ref: @emlx_ref,
      sparse: "emlx_axon"
    }
  end

  @doc "Crucible model surface for EMLX Qwen3."
  def surface(opts \\ []) do
    num_blocks = Keyword.get(opts, :num_blocks, 28)

    metadata = %{
      adapter: :emlx,
      surface_id: :emlx_qwen3,
      capabilities: capabilities(opts),
      emlx_dependency: dependency_pin()
    }

    ModelSurface.new!(:qwen3, Qwen3Surface.nodes(num_blocks), metadata)
  end

  @doc "Capability report for the native EMLX Qwen3 phase-9 surface."
  def capabilities(opts \\ []) do
    num_blocks = Keyword.get(opts, :num_blocks, 28)

    %{
      provider: :emlx_qwen3,
      dependency: dependency_pin(),
      model_family: :qwen3,
      backend: :emlx,
      final_logits: true,
      cache_metadata: true,
      generation_trace: true,
      residual_interventions: true,
      interventions: %{residual: true, head_ablation: false},
      attention_qkv: true,
      attention_scores: true,
      attention_pattern: true,
      attention_z: true,
      mlp_activations: true,
      norm_telemetry: true,
      lazy_tensors: true,
      host_sync_required_for_captures: false,
      capture_groups: @capture_groups,
      activations: activation_claims(num_blocks),
      unsupported: [:head_ablation_intervention]
    }
  end

  @doc """
  Registry-compatible provider/model support metadata for pinned Qwen3 artifacts.

  The return value is a plain map so `crucible_model_registry` can consume it
  without adding an EMLX or Bumblebee runtime dependency.
  """
  def provider_compatibility(opts \\ []) do
    capabilities = capabilities(opts)

    %{
      provider_kind: :emlx_qwen3,
      model_family: :qwen3,
      runtime_profile: :local_emlx,
      supported_signals: [
        :final_logits,
        :generation_step_logits,
        :cache_metadata,
        :residual_stream,
        :attention_q,
        :attention_k,
        :attention_v,
        :attention_scores,
        :attention_weights,
        :head_outputs,
        :mlp_activation,
        :norm_telemetry
      ],
      supported_activations: Map.keys(capabilities.activations),
      supported_capture_groups: capabilities.capture_groups,
      supported_generation_features: [:kv_cache_generation_trace, :cache_metadata],
      supported_active_controls: [:residual_intervention],
      unsupported_active_controls: [:head_ablation],
      unsupported_capture_groups: [],
      unsupported_activations: [],
      metadata: %{
        dependency: dependency_pin(),
        surface_id: :emlx_qwen3,
        host_sync_required_for_captures: false
      }
    }
  end

  @doc """
  Normalizes an `EMLXAxon.Qwen3.Generate.generate_trace/3` result.
  """
  def normalize_generation_trace(trace, opts \\ [])

  def normalize_generation_trace({generated_token_ids, %{trace: trace}}, opts)
      when is_list(generated_token_ids) and is_map(trace) do
    trace
    |> Map.put(:generated_token_ids, generated_token_ids)
    |> normalize_generation_trace(opts)
  end

  def normalize_generation_trace(%{steps: steps} = trace, _opts) when is_list(steps) do
    with :ok <- validate_steps(steps) do
      generated_token_ids = Map.get(trace, :generated_token_ids, Enum.map(steps, & &1.token_id))

      {:ok,
       %{
         provider: :emlx_qwen3,
         generation_success_level: :emlx_qwen3_generation_trace,
         generated_token_ids: generated_token_ids,
         steps: steps,
         trace_metadata: %{
           prompt_length: Map.get(trace, :prompt_length),
           requested_max_new_tokens: Map.get(trace, :requested_max_new_tokens),
           emitted_steps: length(steps),
           cache_offset_source: :emlx_qwen3_generation_trace,
           dependency: dependency_pin()
         }
       }}
    end
  end

  @doc "Drops raw tensor refs from a generation step for public reports."
  def public_step(step) when is_map(step) do
    step
    |> Map.drop([:logits, :model_trace])
    |> Map.update(:cache_metadata, nil, &public_cache_metadata/1)
  end

  @doc "Converts EMLX per-step logits into an ActivationCache."
  def to_activation_cache(%{steps: steps} = trace, opts \\ []) when is_list(steps) do
    {:ok, to_activation_cache!(trace, opts)}
  rescue
    error in [ArgumentError, KeyError] -> {:error, error}
  end

  def to_activation_cache!(%{steps: steps} = trace, opts \\ []) when is_list(steps) do
    logits = generation_logits!(steps)

    spec =
      ActivationSpec.new!(%{
        activation_name: "unembed.hook_logits",
        signal_type: :generation_step_logits,
        axes: [:batch, :pos, :d_vocab],
        capture_mode: :raw,
        requires_raw?: true,
        metadata: %{source: :emlx_qwen3_generation_trace}
      })

    ActivationCache.new!(
      %{"unembed.hook_logits" => logits},
      specs: %{"unembed.hook_logits" => spec},
      model_info: Keyword.get(opts, :model_info, %{}),
      metadata: %{
        source: :emlx_qwen3_generation_trace,
        dependency: dependency_pin(),
        generated_token_ids: Map.get(trace, :generated_token_ids, Enum.map(steps, & &1.token_id))
      }
    )
  end

  defp validate_steps(steps) do
    Enum.reduce_while(steps, :ok, fn step, :ok ->
      cond do
        not is_map(step) ->
          {:halt, {:error, {:invalid_step, step}}}

        not Map.has_key?(step, :token_id) ->
          {:halt, {:error, {:missing_step_field, :token_id}}}

        not Map.has_key?(step, :cache_metadata) ->
          {:halt, {:error, {:missing_step_field, :cache_metadata}}}

        not Map.has_key?(step, :logits_shape) ->
          {:halt, {:error, {:missing_step_field, :logits_shape}}}

        true ->
          {:cont, :ok}
      end
    end)
  end

  defp public_cache_metadata(metadata) when is_map(metadata), do: Map.drop(metadata, [:layers])
  defp public_cache_metadata(metadata), do: metadata

  defp activation_claims(num_blocks) do
    num_blocks
    |> Qwen3Surface.nodes()
    |> Enum.filter(& &1[:activation_name])
    |> Map.new(fn node ->
      {node[:activation_name],
       %{
         signal_type: node[:signal_type],
         layer_index: node[:layer_index],
         axes: node[:axes],
         capture_modes: node[:capture_modes]
       }}
    end)
  end

  defp generation_logits!(steps) do
    steps
    |> Enum.map(&Map.fetch!(&1, :logits))
    |> Nx.stack(axis: 1)
  end
end
