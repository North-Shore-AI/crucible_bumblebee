defmodule CrucibleBumblebee.InterventionRunner do
  @moduledoc """
  Compiles activation interventions against a Bumblebee model surface.

  Current bundled Bumblebee surfaces are read-only for activation internals, so
  active interventions fail closed unless a surface advertises the matching
  `:fuse` or `:gate` operation and the caller supplies a real executor.
  """

  alias CrucibleBumblebee.ModelSurface
  alias CrucibleMechInterp.Intervention

  @doc "Compiles interventions against surface node active-operation support."
  def compile(interventions, %ModelSurface{} = surface) do
    nodes = Enum.map(surface.surface.nodes, &Map.from_struct/1)

    interventions
    |> List.wrap()
    |> Enum.reduce_while({:ok, []}, fn intervention, {:ok, acc} ->
      case Intervention.compile(intervention, nodes) do
        {:ok, compiled} -> {:cont, {:ok, [compiled | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, compiled} -> {:ok, Enum.reverse(compiled)}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Runs interventions with a caller-supplied executor.

  The executor receives `{predict_fun, inputs, compiled_interventions, surface,
  opts}` and must perform the real graph rewrite/rerun. This module does not
  synthesize patched model outputs.
  """
  def run(predict_fun, inputs, interventions, %ModelSurface{} = surface, opts \\ [])
      when is_function(predict_fun, 1) and is_list(opts) do
    with {:ok, compiled} <- compile(interventions, surface),
         {:ok, executor} <- fetch_executor(opts) do
      executor.(predict_fun, inputs, compiled, surface, opts)
    end
  end

  defp fetch_executor(opts) do
    case Keyword.get(opts, :executor) do
      executor when is_function(executor, 5) -> {:ok, executor}
      nil -> {:error, :intervention_executor_required}
      other -> {:error, {:invalid_intervention_executor, other}}
    end
  end
end
