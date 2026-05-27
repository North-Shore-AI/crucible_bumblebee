defmodule CrucibleBumblebeeTest do
  use ExUnit.Case
  doctest CrucibleBumblebee

  test "exposes package version" do
    assert CrucibleBumblebee.version() == "0.1.0"
  end
end
