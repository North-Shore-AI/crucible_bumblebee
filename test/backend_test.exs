defmodule CrucibleBumblebee.BackendTest do
  use ExUnit.Case, async: false

  alias CrucibleBumblebee.Backend

  test "prefer binary returns success and reset restores process-local backend" do
    previous = Nx.default_backend()

    assert {:ok, :binary} = Backend.prefer(:binary)
    assert Nx.default_backend() == {Nx.BinaryBackend, []}
    assert :ok = Backend.reset()
    assert Nx.default_backend() == previous
  end

  test "prefer unavailable backend returns structured diagnostics" do
    assert {:error, {:not_installed, :exla}} = Backend.prefer(:exla_cuda)
  end

  test "auto falls back to binary when accelerator deps are unavailable" do
    assert {:ok, :binary} = Backend.prefer(:auto)
    assert :ok = Backend.reset()
  end
end
