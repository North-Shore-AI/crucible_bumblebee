defmodule CrucibleBumblebee.SurfacePreflight do
  @moduledoc """
  Versioned model-surface pre-flight artifacts.
  """

  alias CrucibleBumblebee.ModelSurface

  def run(%ModelSurface{} = surface, model_info \\ %{}, opts \\ []) do
    opts = preflight_opts(surface, opts)

    with {:ok, payload} <- ModelSurface.preflight(surface, model_info, opts) do
      artifact =
        payload
        |> Map.put(:schema, "crucible_bumblebee.surface_preflight")
        |> Map.put(:version, 1)
        |> Map.put(:surface_id, surface.id)
        |> Map.put(:surface_module, inspect(surface.module))
        |> Map.put(:dependency_fingerprint, dependency_fingerprint(surface))

      if Keyword.get(opts, :write?, true) do
        :ok = write_artifact(surface, artifact, opts)
      end

      {:ok, artifact}
    end
  end

  def ensure_current(%ModelSurface{} = surface, model_info \\ %{}, opts \\ []) do
    fingerprint = dependency_fingerprint(surface)
    path = artifact_path(surface.id, fingerprint, opts)

    case File.read(path) do
      {:ok, body} ->
        with {:ok, artifact} <- Jason.decode(body),
             ^fingerprint <- artifact["dependency_fingerprint"] do
          {:ok, atomize_keys(artifact)}
        else
          _other ->
            if Keyword.get(opts, :auto_write?, false) do
              run(surface, model_info, opts)
            else
              {:error, {:stale_surface_preflight, path}}
            end
        end

      {:error, :enoent} ->
        if Keyword.get(opts, :auto_write?, false) do
          run(surface, model_info, opts)
        else
          {:error, {:missing_surface_preflight, path}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  def write_artifact(%ModelSurface{} = surface, artifact, opts \\ []) do
    path = artifact_path(surface.id, Map.fetch!(artifact, :dependency_fingerprint), opts)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(artifact, pretty: true) <> "\n")
    :ok
  end

  def artifact_path(surface_id, fingerprint, opts \\ []) do
    root = Keyword.get(opts, :root, Path.join(File.cwd!(), "priv/surface_preflight"))
    Path.join(root, "#{surface_id}_#{fingerprint}.json")
  end

  def dependency_fingerprint(%ModelSurface{} = surface) do
    %{
      bumblebee: app_vsn(:bumblebee),
      nx: app_vsn(:nx),
      axon: app_vsn(:axon),
      surface_module: inspect(surface.module),
      surface_id: surface.id,
      surface_options: Map.get(surface.metadata, :surface_options, %{})
    }
    |> :erlang.term_to_binary()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> binary_part(0, 16)
  end

  defp app_vsn(app) do
    case Application.spec(app, :vsn) do
      nil -> "unknown"
      vsn -> List.to_string(vsn)
    end
  end

  defp preflight_opts(%ModelSurface{} = surface, opts) do
    surface_opts =
      surface.metadata
      |> Map.get(:surface_options, %{})
      |> Map.to_list()

    Keyword.merge(surface_opts, opts)
  end

  defp atomize_keys(value) when is_map(value) do
    Map.new(value, fn
      {key, value} when is_binary(key) -> {String.to_atom(key), atomize_keys(value)}
      {key, value} -> {key, atomize_keys(value)}
    end)
  end

  defp atomize_keys(value) when is_list(value), do: Enum.map(value, &atomize_keys/1)
  defp atomize_keys(value), do: value
end
