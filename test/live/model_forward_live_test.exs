defmodule CrucibleBumblebee.Live.ModelForwardLiveTest do
  use ExUnit.Case, async: false

  @moduletag :live_cpu_heavy

  test "live model forward is opt-in" do
    assert System.get_env("CRUCIBLE_BUMBLEBEE_LIVE") in ["1", "true"]
  end
end
