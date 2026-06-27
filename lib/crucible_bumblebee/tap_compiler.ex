defmodule CrucibleBumblebee.TapCompiler do
  @moduledoc """
  Compiles Crucible tap plans against Bumblebee model surfaces.
  """

  alias CrucibleBumblebee.{CompiledTapPlan, ModelSurface, SurfacePreflight}
  alias CrucibleTap.{PlanCompiler, TapPlan}

  def compile(%TapPlan{} = plan, %ModelSurface{} = model_surface, opts \\ []) do
    with {:ok, preflight} <-
           SurfacePreflight.ensure_current(model_surface, %{},
             auto_write?: Keyword.get(opts, :auto_write_preflight?, true)
           ),
         {:ok, compiled} <- PlanCompiler.compile(plan, model_surface.surface),
         :ok <- verify_preflight_bindings(compiled.matched, preflight) do
      {:ok,
       %CompiledTapPlan{
         tap_plan_id: plan.plan_id,
         global_layer_options:
           model_surface
           |> ModelSurface.output_options(compiled)
           |> merge_options(global_layer_options(plan, compiled.matched)),
         hook_names: hook_names(compiled.matched),
         matched: compiled.matched,
         unsupported_optional: compiled.report.unsupported_optional,
         report: compiled.report,
         metadata: %{
           model_family: model_surface.family,
           surface_id: model_surface.id,
           dependency_fingerprint: preflight.dependency_fingerprint
         }
       }}
    end
  end

  defp verify_preflight_bindings(matched, preflight) do
    available_nodes = Map.get(preflight, :nodes, [])

    missing =
      matched
      |> Enum.map(& &1.metadata[:layer_name])
      |> Enum.reject(&is_nil/1)
      |> Enum.reject(&(&1 in available_nodes))

    if missing == [], do: :ok, else: {:error, {:preflight_missing_nodes, missing}}
  end

  defp global_layer_options(%TapPlan{} = plan, matched) do
    signal_types = Enum.map(plan.specs, & &1.signal_spec.signal_type)
    activation_names = activation_names(plan, matched)

    []
    |> maybe_put(:output_hidden_states, Enum.any?(signal_types, &hidden_state_type?/1))
    |> maybe_put(
      :output_attentions,
      Enum.any?(signal_types, &(&1 in [:attention_maps, :attention_weights])) or
        Enum.any?(activation_names, &String.ends_with?(&1, ".attn.hook_pattern"))
    )
    |> maybe_put(
      :output_attention_qkv,
      Enum.any?(signal_types, &attention_activation_type?/1) or
        Enum.any?(activation_names, &attention_activation_name?/1)
    )
    |> maybe_put(
      :output_attention_scores,
      Enum.any?(signal_types, &(&1 == :attention_scores)) or
        Enum.any?(activation_names, &String.ends_with?(&1, ".attn.hook_attn_scores"))
    )
    |> maybe_put(
      :output_mlp_activations,
      Enum.any?(signal_types, &mlp_activation_type?/1) or
        Enum.any?(activation_names, &mlp_activation_name?/1)
    )
    |> maybe_put(
      :output_residual_streams,
      residual_stream_output_requested?(plan, activation_names)
    )
    |> maybe_put(
      :output_norm_telemetry,
      Enum.any?(signal_types, &(&1 == :norm_telemetry)) or
        Enum.any?(activation_names, &norm_telemetry_activation_name?/1)
    )
  end

  defp hidden_state_type?(type) do
    type in [:embeddings, :early_residuals, :middle_residuals, :late_residuals, :layer_trajectory]
  end

  defp attention_activation_type?(type) do
    type in [:attention_q, :attention_k, :attention_v, :head_outputs]
  end

  defp mlp_activation_type?(type) do
    type in [:mlp_activation, :mlp_gates]
  end

  defp activation_names(%TapPlan{} = plan, matched) do
    plan_names =
      plan.specs
      |> Enum.flat_map(fn spec ->
        [
          spec.selector.activation_name,
          spec.metadata[:activation_name]
        ]
      end)

    matched_names = Enum.map(matched, & &1.metadata[:activation_name])

    (plan_names ++ matched_names)
    |> Enum.filter(&is_binary/1)
    |> Enum.uniq()
  end

  defp attention_activation_name?(name) do
    String.ends_with?(name, ".attn.hook_q") or
      String.ends_with?(name, ".attn.hook_k") or
      String.ends_with?(name, ".attn.hook_v") or
      String.ends_with?(name, ".attn.hook_z") or
      String.ends_with?(name, ".hook_attn_out")
  end

  defp mlp_activation_name?(name) do
    String.ends_with?(name, ".mlp.hook_pre") or
      String.ends_with?(name, ".mlp.hook_post") or
      String.ends_with?(name, ".hook_mlp_out")
  end

  defp residual_stream_output_requested?(%TapPlan{} = plan, activation_names) do
    Enum.any?(activation_names, &residual_stream_activation_name?/1) or
      Enum.any?(plan.specs, fn spec ->
        spec.signal_spec.signal_type == :residual_stream and
          case spec_activation_name(spec) do
            nil -> true
            activation_name -> residual_stream_activation_name?(activation_name)
          end
      end)
  end

  defp norm_telemetry_activation_name?(name) do
    name in ["ln_final.hook_scale", "ln_final.hook_normalized"]
  end

  defp residual_stream_activation_name?(name), do: String.contains?(name, "hook_resid")

  defp spec_activation_name(spec) do
    spec.selector.activation_name || spec.metadata[:activation_name]
  end

  defp maybe_put(opts, key, true), do: Keyword.put(opts, key, true)
  defp maybe_put(opts, _key, false), do: opts

  defp merge_options(left, right), do: Keyword.merge(right, left)

  defp hook_names(matched) do
    matched
    |> Enum.map(& &1.metadata[:layer_name])
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
  end
end
