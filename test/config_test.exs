defmodule CrucibleBumblebee.ConfigTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.Config

  test "resolves runtime config through an injected environment reader" do
    env_reader = fn
      "CRUCIBLE_BUMBLEBEE_LIVE" -> "true"
      "CRUCIBLE_ARTIFACT_ROOT" -> "tmp/custom_root"
      "CRUCIBLE_TRACE_DIR" -> "tmp/custom_traces"
      "CRUCIBLE_BUMBLEBEE_BACKEND" -> "binary"
      "CRUCIBLE_BUMBLEBEE_MAX_NEW_TOKENS" -> "4"
      _name -> nil
    end

    assert Config.live_enabled?(env_reader)
    assert Config.artifact_root([], env_reader) == "tmp/custom_root"
    assert Config.trace_dir([], env_reader) == "tmp/custom_traces"
    assert Config.backend("auto", env_reader) == "binary"
    assert Config.integer_env("CRUCIBLE_BUMBLEBEE_MAX_NEW_TOKENS", 8, env_reader) == 4
  end

  test "explicit options override environment values" do
    env_reader = fn
      "CRUCIBLE_ARTIFACT_ROOT" -> "tmp/env_root"
      "CRUCIBLE_TRACE_DIR" -> "tmp/env_traces"
      _name -> nil
    end

    assert Config.artifact_root([root: "tmp/explicit_root"], env_reader) == "tmp/explicit_root"

    assert Config.trace_dir([trace_dir: "tmp/explicit_traces"], env_reader) ==
             "tmp/explicit_traces"
  end
end
