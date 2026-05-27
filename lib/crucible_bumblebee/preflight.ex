defmodule CrucibleBumblebee.Preflight do
  @moduledoc """
  Native runner preflight checks.
  """

  alias CrucibleBumblebee.{ModelBundle, ModelSurface, Qwen3Surface}
  alias CrucibleBumblebee.Surfaces.Gpt2CausalLMSurface
  alias CrucibleTap.TapPlan

  def run(%ModelBundle{} = bundle, %TapPlan{} = tap_plan) do
    with true <- not is_nil(bundle.tokenizer) || {:error, :tokenizer_not_loaded},
         true <- not is_nil(bundle.params) || {:error, :model_params_not_loaded},
         {:ok, surface} <- surface_for_bundle(bundle),
         {:ok, surface_preflight} <- ModelSurface.preflight(surface, model_info(bundle), []),
         {:ok, compiled, report} <-
           Crucible.CapabilityReport.negotiate(tap_plan, surface.surface,
             provider_kind: :elixir_bumblebee,
             model_id: bundle.model_id,
             model_family: bundle.model_family,
             backend: bundle.backend,
             resource_budget: resource_budget()
           ) do
      {:ok,
       %{
         surface: surface,
         surface_preflight: surface_preflight,
         compiled_tap_plan: compiled,
         capability_report: report
       }}
    end
  end

  def run!(%ModelBundle{} = bundle, %TapPlan{} = tap_plan) do
    case run(bundle, tap_plan) do
      {:ok, preflight} -> preflight
      {:error, reason} -> raise "preflight failed: #{inspect(reason)}"
      false -> raise "preflight failed"
      other -> raise "preflight failed: #{inspect(other)}"
    end
  end

  def surface_for_bundle(%ModelBundle{model_family: :gpt2}),
    do: {:ok, Gpt2CausalLMSurface.surface()}

  def surface_for_bundle(%ModelBundle{model_family: :qwen3}), do: {:ok, Qwen3Surface.surface()}

  def surface_for_bundle(%ModelBundle{model_family: family}) when family in [:distilbert],
    do: {:ok, final_logits_surface(family)}

  def surface_for_bundle(%ModelBundle{model_family: family}),
    do: {:error, {:unsupported_live_surface_family, family}}

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

  defp model_info(%ModelBundle{} = bundle) do
    %{
      model_id: bundle.model_id,
      tokenizer_id: bundle.tokenizer_id,
      model_family: bundle.model_family,
      backend: bundle.backend,
      revision: bundle.revision,
      spec: bundle.spec
    }
  end

  defp final_logits_surface(family) do
    ModelSurface.new!(family, final_logits_nodes(), %{
      surface_id: :"#{family}_final_logits",
      capabilities: %{
        final_logits: true,
        hidden_state: false,
        attention_weights: false,
        generation_step_logits: false,
        active_injection: false
      }
    })
  end

  defp final_logits_nodes do
    [
      [
        id: "final_logits",
        signal_type: :final_logits,
        layer_name: "final_logits",
        layer_index: :final,
        operations: [:read, :route_on],
        capture_modes: [:summary]
      ]
    ]
  end
end
