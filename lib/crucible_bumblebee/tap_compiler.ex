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
           |> merge_options(global_layer_options(plan)),
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

  defp global_layer_options(%TapPlan{} = plan) do
    signal_types = Enum.map(plan.specs, & &1.signal_spec.signal_type)

    []
    |> maybe_put(:output_hidden_states, Enum.any?(signal_types, &hidden_state_type?/1))
    |> maybe_put(:output_attentions, :attention_maps in signal_types)
  end

  defp hidden_state_type?(type) do
    type in [:embeddings, :early_residuals, :middle_residuals, :late_residuals, :layer_trajectory]
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
