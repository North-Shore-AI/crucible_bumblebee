defmodule CrucibleBumblebee.GenerationRunner do
  @moduledoc """
  Boundary for Bumblebee/Nx generation serving invocation.
  """

  alias CrucibleBumblebee.ModelSurface
  alias CruciblePolicy.SteeringPlan

  def generate(serving, input, steering_plan, surface, opts \\ [])

  def generate(%Nx.Serving{} = serving, input, steering_plan, %ModelSurface{} = surface, opts) do
    with :ok <- steering_available?(steering_plan, surface, opts) do
      {:ok, Nx.Serving.run(serving, input)}
    end
  end

  def generate(generate_fun, input, steering_plan, %ModelSurface{} = surface, opts)
      when is_function(generate_fun, 2) do
    with :ok <- custom_loop_available?(opts),
         :ok <- steering_available?(steering_plan, surface, Keyword.put(opts, :custom_loop, true)) do
      {:ok, generate_fun.(input, steering_plan)}
    end
  end

  def generate(generate_fun, input, _steering_plan, %ModelSurface{}, _opts)
      when is_function(generate_fun, 1) do
    {:ok, generate_fun.(input)}
  end

  defp steering_available?(nil, _surface, _opts), do: :ok

  defp steering_available?(%SteeringPlan{mode: :in_graph}, %ModelSurface{} = surface, _opts) do
    if capability?(surface, :in_graph_steering) do
      :ok
    else
      {:error, :steering_surface_unavailable}
    end
  end

  defp steering_available?(%SteeringPlan{}, %ModelSurface{} = surface, opts) do
    cond do
      capability?(surface, :logits_processors) -> :ok
      Keyword.get(opts, :custom_loop, false) -> :ok
      true -> {:error, :steering_surface_unavailable}
    end
  end

  defp custom_loop_available?(opts) do
    if Keyword.get(opts, :custom_loop, false), do: :ok, else: {:error, :custom_loop_required}
  end

  defp capability?(%ModelSurface{capabilities: capabilities}, capability) do
    Map.get(capabilities, capability) in [true, "true", 1]
  end
end
