defmodule CrucibleBumblebee.Preflight do
  @moduledoc """
  Bounded V4 preflight checks for the curated native runner.
  """

  alias CrucibleBumblebee.{ModelBundle, Surfaces.TinyGPT2Surface}
  alias CrucibleTap.TapPlan

  def run!(%ModelBundle{} = bundle, %TapPlan{} = tap_plan) do
    surface = TinyGPT2Surface.surface()

    with true <-
           bundle.model_id in [
             CrucibleBumblebee.ModelLoader.default_model_id(),
             System.get_env("CRUCIBLE_BUMBLEBEE_MODEL_ID")
           ] || {:error, :unexpected_model_id},
         true <- not is_nil(bundle.tokenizer) || {:error, :tokenizer_not_loaded},
         true <- not is_nil(bundle.params) || {:error, :model_params_not_loaded},
         {:ok, compiled, report} <-
           Crucible.CapabilityReport.negotiate(tap_plan, surface.surface,
             provider_kind: :elixir_bumblebee,
             model_id: bundle.model_id,
             model_family: bundle.model_family,
             backend: bundle.backend,
             resource_budget: resource_budget()
           ) do
      %{surface: surface, compiled_tap_plan: compiled, capability_report: report}
    else
      {:error, reason} -> raise "preflight failed: #{inspect(reason)}"
      false -> raise "preflight failed"
      other -> raise "preflight failed: #{inspect(other)}"
    end
  end

  def resource_budget do
    %Crucible.CapabilityReport.ResourceBudget{
      max_extra_forward_passes: 0,
      max_parallel_kv_caches: 1,
      supports_token_callback?: false,
      supports_auxiliary_forward?: false,
      supports_active_injection?: false,
      estimated_vram_multiplier: 1.0
    }
  end
end
