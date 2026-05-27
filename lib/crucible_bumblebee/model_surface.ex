defmodule CrucibleBumblebee.ModelSurface do
  @moduledoc """
  Behaviour and wrapper for model-family-specific Bumblebee surfaces.

  Runners consume this behaviour through the wrapper struct. Model-specific
  node names, output options, and logit-lens params paths live in surface
  modules, not in the reusable runner code.
  """

  alias CrucibleTap.Surface

  @callback id() :: atom()
  @callback model_family() :: atom()
  @callback capabilities(keyword()) :: map()
  @callback output_options(CrucibleTap.CompiledPlan.t() | map()) :: keyword()
  @callback preflight(model_info :: map(), opts :: keyword()) :: {:ok, map()} | {:error, term()}
  @callback logit_lens_access(model_info :: map(), params :: map()) ::
              {:ok, map()} | {:error, :unsupported | term()}

  @derive Jason.Encoder
  defstruct id: nil,
            module: nil,
            family: nil,
            surface: nil,
            capabilities: %{},
            metadata: %{}

  @type t :: %__MODULE__{}

  def from_module!(module, opts \\ []) when is_atom(module) do
    ensure_surface_module!(module)

    id = module.id()
    family = module.model_family()
    nodes = Keyword.get(opts, :nodes) || module.nodes(Keyword.get(opts, :num_blocks, 1))

    metadata =
      opts
      |> Keyword.get(:metadata, %{})
      |> Map.put_new(:surface_options, Map.new(Keyword.take(opts, [:num_blocks])))

    capabilities = module.capabilities(opts)

    %__MODULE__{
      id: id,
      module: module,
      family: family,
      surface:
        Surface.new!(
          adapter: :bumblebee,
          model_family: family,
          nodes: nodes,
          metadata: Map.merge(metadata, %{surface_id: id, surface_module: inspect(module)})
        ),
      capabilities: capabilities,
      metadata: metadata
    }
  end

  def new!(family, nodes, metadata \\ %{}) do
    %__MODULE__{
      id: Map.get(metadata, :surface_id, family),
      family: family,
      surface:
        Surface.new!(
          adapter: :bumblebee,
          model_family: family,
          nodes: nodes,
          metadata: metadata
        ),
      capabilities: Map.get(metadata, :capabilities, %{}),
      metadata: metadata
    }
  end

  def output_options(%__MODULE__{module: module}, compiled_plan) when is_atom(module) do
    module.output_options(compiled_plan)
  end

  def output_options(%__MODULE__{}, compiled_plan) do
    Map.get(compiled_plan, :global_layer_options, [])
  end

  def preflight(surface, model_info, opts \\ [])

  def preflight(%__MODULE__{module: module}, model_info, opts) when is_atom(module) do
    module.preflight(model_info, opts)
  end

  def preflight(%__MODULE__{} = surface, _model_info, _opts) do
    {:ok,
     %{
       surface_id: surface.id,
       model_family: surface.family,
       capabilities: surface.capabilities,
       nodes: Enum.map(surface.surface.nodes, & &1.id),
       unsupported: []
     }}
  end

  def logit_lens_access(%__MODULE__{module: module}, model_info, params) when is_atom(module) do
    module.logit_lens_access(model_info, params)
  end

  def logit_lens_access(%__MODULE__{}, _model_info, _params), do: {:error, :unsupported}

  defp ensure_surface_module!(module) do
    for {function, arity} <- [
          id: 0,
          model_family: 0,
          capabilities: 1,
          output_options: 1,
          preflight: 2,
          logit_lens_access: 2,
          nodes: 1
        ] do
      unless function_exported?(module, function, arity) do
        raise ArgumentError,
              "#{inspect(module)} is missing ModelSurface callback #{function}/#{arity}"
      end
    end
  end
end
