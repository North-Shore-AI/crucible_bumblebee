defmodule CrucibleBumblebee.ModelLoaderTest do
  use ExUnit.Case, async: false

  alias CrucibleBumblebee.ModelLoader
  alias CrucibleBumblebee.ModelLoader.Options

  setup do
    keys = ~w(
      CRUCIBLE_BUMBLEBEE_MODEL_ID
      CRUCIBLE_BUMBLEBEE_TOKENIZER_ID
      CRUCIBLE_BUMBLEBEE_BACKEND
      CRUCIBLE_BACKEND
      CRUCIBLE_BUMBLEBEE_REVISION
      CRUCIBLE_BUMBLEBEE_OFFLINE
      CRUCIBLE_HF_OFFLINE
      CRUCIBLE_MODEL_CACHE_DIR
      CRUCIBLE_PROMPT
      CRUCIBLE_BUMBLEBEE_MAX_NEW_TOKENS
      CRUCIBLE_BUMBLEBEE_SEED
      CRUCIBLE_ARTIFACT_ROOT
      CRUCIBLE_BUMBLEBEE_DIAGNOSTIC_PATH
    )

    previous = Map.new(keys, &{&1, System.get_env(&1)})
    on_exit(fn -> Enum.each(previous, fn {key, value} -> restore_env(key, value) end) end)
    :ok
  end

  test "resolves model loader options from explicit values" do
    options =
      Options.new(
        model_id: "gpt2",
        tokenizer_id: "gpt2-tokenizer",
        revision: "main",
        backend: "exla-cpu",
        offline?: true,
        cache_dir: "/tmp/hf-cache",
        prompt: "Hello",
        max_new_tokens: 4,
        seed: 7
      )

    assert options.model_id == "gpt2"
    assert options.tokenizer_id == "gpt2-tokenizer"
    assert options.backend == :exla_cpu
    assert options.offline?

    assert Options.repository(options) ==
             {:hf, "gpt2", revision: "main", cache_dir: "/tmp/hf-cache", offline: true}
  end

  test "resolves model loader options from environment" do
    System.put_env("CRUCIBLE_BUMBLEBEE_MODEL_ID", "distilgpt2")
    System.put_env("CRUCIBLE_BUMBLEBEE_BACKEND", "binary")
    System.put_env("CRUCIBLE_BUMBLEBEE_OFFLINE", "true")
    System.put_env("CRUCIBLE_BUMBLEBEE_MAX_NEW_TOKENS", "3")

    options = Options.from_env()

    assert options.model_id == "distilgpt2"
    assert options.tokenizer_id == "distilgpt2"
    assert options.backend == :binary
    assert options.offline?
    assert options.max_new_tokens == 3
  end

  test "reports offline cache miss without touching the network" do
    cache_dir =
      Path.join(System.tmp_dir!(), "missing_hf_cache_#{System.unique_integer([:positive])}")

    on_exit(fn -> File.rm_rf!(cache_dir) end)
    options = Options.new(model_id: "gpt2", offline?: true, cache_dir: cache_dir)

    assert Options.cache_status(options) == :cache_dir_missing

    diagnostic_path =
      Path.join(System.tmp_dir!(), "loader_diagnostic_#{System.unique_integer([:positive])}.json")

    on_exit(fn -> File.rm(diagnostic_path) end)
    diagnostic = %{ok: false, model_id: "gpt2", reason: :cache_dir_missing}

    assert ModelLoader.write_diagnostic!(diagnostic_path, diagnostic) == diagnostic_path
    assert Jason.decode!(File.read!(diagnostic_path))["reason"] == "cache_dir_missing"
  end

  test "writes success diagnostics when requested" do
    diagnostic_path =
      Path.join(System.tmp_dir!(), "loader_success_#{System.unique_integer([:positive])}.json")

    on_exit(fn -> File.rm(diagnostic_path) end)

    options = Options.new(model_id: "gpt2", diagnostic_path: diagnostic_path)
    diagnostic = %{ok: true, model_id: options.model_id, backend: :binary}

    ModelLoader.write_diagnostic!(diagnostic_path, diagnostic)

    assert %{"ok" => true, "model_id" => "gpt2", "backend" => "binary"} =
             Jason.decode!(File.read!(diagnostic_path))
  end

  test "returns structured diagnostics for unsupported model families" do
    assert {:error, diagnostic} =
             ModelLoader.load(model_id: "facebook/opt-125m", backend: :binary, offline?: true)

    assert diagnostic.ok == false
    assert diagnostic.reason == {:unsupported_model_family, "facebook/opt-125m"}
  end

  defp restore_env(key, nil), do: System.delete_env(key)
  defp restore_env(key, value), do: System.put_env(key, value)
end
