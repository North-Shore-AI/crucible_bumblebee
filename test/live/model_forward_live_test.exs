defmodule CrucibleBumblebee.Live.ModelForwardLiveTest do
  use ExUnit.Case, async: false

  @moduletag :live_cpu_heavy

  test "live model forward is opt-in" do
    assert CrucibleBumblebee.Config.live_enabled?()
  end
end
