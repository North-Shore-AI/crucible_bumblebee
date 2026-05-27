defmodule CrucibleBumblebee.ModelSurface do
  @moduledoc """
  Bumblebee-facing model surface wrapper.
  """

  alias CrucibleTap.Surface

  @derive Jason.Encoder
  defstruct family: nil, surface: nil, metadata: %{}

  @type t :: %__MODULE__{}

  def new!(family, nodes, metadata \\ %{}) do
    %__MODULE__{
      family: family,
      surface:
        Surface.new!(
          adapter: :bumblebee,
          model_family: family,
          nodes: nodes,
          metadata: metadata
        ),
      metadata: metadata
    }
  end
end
