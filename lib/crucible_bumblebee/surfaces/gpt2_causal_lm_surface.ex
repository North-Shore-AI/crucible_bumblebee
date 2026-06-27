defmodule CrucibleBumblebee.Surfaces.Gpt2CausalLMSurface do
  @moduledoc """
  Final-logits surface for GPT-2-family causal language models.

  This intentionally describes the native signal surface that the current
  Bumblebee runner can prove across GPT-2-family models. Internals such as
  hidden states and attention weights remain explicit degraded capabilities
  until a later internals pass captures them from the Axon/Bumblebee graph.
  """

  @behaviour CrucibleBumblebee.ModelSurface

  alias CrucibleBumblebee.ModelSurface

  def surface(opts \\ []), do: ModelSurface.from_module!(__MODULE__, opts)

  @impl true
  def id, do: :gpt2_causal_lm

  @impl true
  def model_family, do: :gpt2

  @impl true
  def capabilities(_opts \\ []) do
    %{
      final_logits: true,
      hidden_state: false,
      attention_weights: false,
      generation_step_logits: false,
      active_injection: false,
      token_callback: false,
      auxiliary_forward_pass: false
    }
  end

  @impl true
  def output_options(_compiled_plan), do: []

  @impl true
  def preflight(model_info, _opts) do
    with :ok <- validate_family(model_info) do
      {:ok,
       %{
         surface_id: id(),
         model_family: model_family(),
         nodes: Enum.map(nodes(1), & &1[:id]),
         post_processing_extractors: [:final_logits],
         unsupported: [
           :hidden_state,
           :attention_weights,
           :generation_step_logits,
           :active_injection
         ]
       }}
    end
  end

  @impl true
  def logit_lens_access(_model_info, _params), do: {:error, :unsupported}

  def nodes(_num_blocks \\ 1) do
    metadata = CrucibleBumblebee.ActivationMapper.final_logits()

    [
      [
        id: "final_logits",
        signal_type: :final_logits,
        activation_name: Map.get(metadata, :activation_name),
        axes: Map.get(metadata, :axes),
        layer_name: "final_logits",
        layer_index: :final,
        operations: [:read, :route_on],
        capture_modes: [:summary],
        metadata: metadata
      ]
    ]
  end

  defp validate_family(%{model_family: :gpt2}), do: :ok
  defp validate_family(%{"model_family" => :gpt2}), do: :ok
  defp validate_family(%{"model_family" => "gpt2"}), do: :ok
  defp validate_family(%{model_family: family}), do: {:error, {:surface_family_mismatch, family}}

  defp validate_family(%{"model_family" => family}),
    do: {:error, {:surface_family_mismatch, family}}

  defp validate_family(_model_info), do: {:error, :missing_model_family}
end
