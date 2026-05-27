defmodule CrucibleBumblebee.ModelLoader.LoadError do
  @moduledoc "Structured model-loader exception."

  defexception [:diagnostic]

  @impl true
  def message(%__MODULE__{diagnostic: diagnostic}) do
    "model load failed: #{inspect(diagnostic)}"
  end
end
