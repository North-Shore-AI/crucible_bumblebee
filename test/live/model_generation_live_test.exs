defmodule CrucibleBumblebee.Live.ModelGenerationLiveTest do
  use ExUnit.Case, async: false

  @moduletag :live_cpu_heavy

  test "live model generation is opt-in" do
    assert CrucibleBumblebee.Config.live_enabled?()
  end
end
