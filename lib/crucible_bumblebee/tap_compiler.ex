defmodule CrucibleBumblebee.TapCompiler do
  @moduledoc """
  Compiles Crucible tap plans against Bumblebee model surfaces.
  """

  alias CrucibleBumblebee.{CompiledTapPlan, ModelSurface}
  alias CrucibleTap.{PlanCompiler, TapPlan}

  def compile(%TapPlan{} = plan, %ModelSurface{} = model_surface) do
    with {:ok, compiled} <- PlanCompiler.compile(plan, model_surface.surface) do
      {:ok,
       %CompiledTapPlan{
         tap_plan_id: plan.plan_id,
         global_layer_options: global_layer_options(plan),
         hook_names: hook_names(compiled.matched),
         matched: compiled.matched,
         unsupported_optional: compiled.report.unsupported_optional,
         report: compiled.report,
         metadata: %{model_family: model_surface.family}
       }}
    end
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

  defp hook_names(matched) do
    matched
    |> Enum.map(& &1.metadata[:layer_name])
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
  end
end
