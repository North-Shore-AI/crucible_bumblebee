defmodule CrucibleBumblebee.Backend do
  @moduledoc """
  Structured backend preference diagnostics for one-shot runs and tests.
  """

  @process_previous_key {__MODULE__, :previous_backend}

  def prefer(:auto) do
    [:exla_cuda, :exla_cpu, :torchx_cuda, :torchx_cpu, :binary]
    |> Enum.reduce_while([], fn backend, attempts ->
      case prefer(backend) do
        {:ok, selected} -> {:halt, {:ok, selected}}
        {:error, reason} -> {:cont, [{backend, reason} | attempts]}
      end
    end)
    |> case do
      {:ok, selected} -> {:ok, selected}
      attempts -> {:error, {:no_available_backend, Enum.reverse(attempts)}}
    end
  end

  def prefer(:binary), do: set_backend(:binary, Nx.BinaryBackend)
  def prefer(:exla_cpu), do: prefer_exla(:exla_cpu, :cpu)
  def prefer(:exla_cuda), do: prefer_exla(:exla_cuda, :cuda)
  def prefer(:exla_rocm), do: prefer_exla(:exla_rocm, :rocm)
  def prefer(:torchx_cpu), do: prefer_torchx(:torchx_cpu, :cpu)
  def prefer(:torchx_cuda), do: prefer_torchx(:torchx_cuda, :cuda)
  def prefer(other), do: {:error, {:unknown_backend, other}}

  def reset do
    case Process.get(@process_previous_key) do
      nil ->
        :ok

      previous ->
        Nx.default_backend(previous)
        Process.delete(@process_previous_key)
        :ok
    end
  end

  defp prefer_exla(selected, device) do
    if Code.ensure_loaded?(EXLA.Backend) do
      set_backend(selected, {EXLA.Backend, device: device})
    else
      {:error, {:not_installed, :exla}}
    end
  rescue
    error -> {:error, {:no_device, selected, Exception.message(error)}}
  end

  defp prefer_torchx(selected, device) do
    if Code.ensure_loaded?(Torchx.Backend) do
      set_backend(selected, {Torchx.Backend, device: device})
    else
      {:error, {:not_installed, :torchx}}
    end
  rescue
    error -> {:error, {:no_device, selected, Exception.message(error)}}
  end

  defp set_backend(selected, backend) do
    if is_nil(Process.get(@process_previous_key)) do
      Process.put(@process_previous_key, Nx.default_backend())
    end

    Nx.default_backend(backend)
    {:ok, selected}
  rescue
    error -> {:error, {:no_device, selected, Exception.message(error)}}
  end
end
