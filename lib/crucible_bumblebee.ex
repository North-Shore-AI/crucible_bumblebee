defmodule CrucibleBumblebee do
  @moduledoc """
  Bumblebee, Axon, and Nx adapter layer for Crucible tap plans and traces.

  This package compiles model-independent tap requests into Bumblebee model
  outputs, Axon hooks, Nx serving hooks, and decode-time logits processors.
  """

  @version Mix.Project.config()[:version]

  @doc "Returns the package version."
  def version, do: @version
end
