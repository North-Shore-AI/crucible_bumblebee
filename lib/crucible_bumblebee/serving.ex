defmodule CrucibleBumblebee.Serving do
  @moduledoc """
  Data contract for a future Nx.Serving-backed Crucible runtime.
  """

  @derive Jason.Encoder
  defstruct serving_ref: nil,
            model_id: nil,
            surface: nil,
            compiled_taps: nil,
            predict_fun: nil,
            metadata: %{}
end
