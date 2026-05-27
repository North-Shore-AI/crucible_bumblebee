defmodule CrucibleBumblebee.Serving do
  @moduledoc """
  Data contract for a future Nx.Serving-backed Crucible runtime.
  """

  @derive Jason.Encoder
  defstruct serving_ref: nil, model_ref: nil, surface: nil, metadata: %{}
end
