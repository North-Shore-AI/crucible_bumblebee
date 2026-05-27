defmodule CrucibleBumblebee.AxonHooks do
  @moduledoc """
  Small hook helpers used by compiled tap plans.
  """

  def capture_to(pid, label) when is_pid(pid) do
    fn value ->
      send(pid, {:crucible_bumblebee_hook, label, value})
      value
    end
  end
end
