defmodule CrucibleBumblebee.Surfaces.TinyGPT2Surface do
  @moduledoc """
  Curated V5 control surface for `hf-internal-testing/tiny-random-gpt2`.
  """

  @behaviour CrucibleBumblebee.ModelSurface

  alias CrucibleBumblebee.ModelSurface

  def surface(opts \\ []), do: ModelSurface.from_module!(__MODULE__, opts)

  @impl true
  def id, do: :tiny_gpt2

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
  def preflight(_model_info, _opts) do
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

  @impl true
  def logit_lens_access(_model_info, _params), do: {:error, :unsupported}

  def nodes(_num_blocks \\ 1) do
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
