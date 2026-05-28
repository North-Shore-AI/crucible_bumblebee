defmodule CrucibleBumblebee.Config do
  @moduledoc """
  Package-local runtime configuration boundary for Crucible Bumblebee.

  Domain modules must receive explicit options or call this module instead of
  reading OS environment variables directly.
  """

  @default_artifact_root "tmp/crucible_v5"
  @default_model_id "hf-internal-testing/tiny-random-gpt2"
  @default_prompt "Hi"

  @type env_reader :: (String.t() -> String.t() | nil)

  @spec env(String.t()) :: String.t() | nil
  def env(name), do: System.get_env(name)

  @spec live_enabled?(env_reader()) :: boolean()
  def live_enabled?(env_reader \\ &env/1) when is_function(env_reader, 1) do
    truthy?(env_reader.("CRUCIBLE_BUMBLEBEE_LIVE"))
  end

  @spec artifact_root(keyword(), env_reader()) :: Path.t()
  def artifact_root(opts \\ [], env_reader \\ &env/1) when is_function(env_reader, 1) do
    Keyword.get(opts, :root) ||
      Keyword.get(opts, :artifact_root) ||
      env_reader.("CRUCIBLE_ARTIFACT_ROOT") ||
      @default_artifact_root
  end

  @spec trace_dir(keyword(), env_reader()) :: Path.t() | nil
  def trace_dir(opts \\ [], env_reader \\ &env/1) when is_function(env_reader, 1) do
    Keyword.get(opts, :trace_dir) || env_reader.("CRUCIBLE_TRACE_DIR")
  end

  @spec backend(String.t(), env_reader()) :: String.t()
  def backend(default \\ "auto", env_reader \\ &env/1) when is_function(env_reader, 1) do
    env_reader.("CRUCIBLE_BUMBLEBEE_BACKEND") ||
      env_reader.("CRUCIBLE_BACKEND") ||
      default
  end

  @spec model_loader_attrs(keyword(), env_reader()) :: keyword()
  def model_loader_attrs(overrides \\ [], env_reader \\ &env/1) when is_function(env_reader, 1) do
    [
      model_id: env_reader.("CRUCIBLE_BUMBLEBEE_MODEL_ID") || @default_model_id,
      tokenizer_id: env_reader.("CRUCIBLE_BUMBLEBEE_TOKENIZER_ID"),
      revision: env_reader.("CRUCIBLE_BUMBLEBEE_REVISION"),
      backend: backend("auto", env_reader),
      offline?:
        truthy?(
          env_reader.("CRUCIBLE_BUMBLEBEE_OFFLINE") ||
            env_reader.("CRUCIBLE_HF_OFFLINE")
        ),
      cache_dir: env_reader.("CRUCIBLE_MODEL_CACHE_DIR"),
      prompt: env_reader.("CRUCIBLE_PROMPT") || @default_prompt,
      max_new_tokens: integer_env("CRUCIBLE_BUMBLEBEE_MAX_NEW_TOKENS", 8, env_reader),
      seed: integer_env("CRUCIBLE_BUMBLEBEE_SEED", nil, env_reader),
      artifact_root: artifact_root([], env_reader),
      diagnostic_path: env_reader.("CRUCIBLE_BUMBLEBEE_DIAGNOSTIC_PATH")
    ]
    |> Keyword.merge(overrides)
  end

  @spec integer_env(String.t(), integer() | nil, env_reader()) :: integer() | nil
  def integer_env(name, default \\ nil, env_reader \\ &env/1) when is_function(env_reader, 1) do
    case env_reader.(name) do
      nil ->
        default

      value ->
        case Integer.parse(value) do
          {integer, ""} -> integer
          _other -> default
        end
    end
  end

  @spec truthy?(term()) :: boolean()
  def truthy?(value), do: value in [true, "1", "true", "TRUE", "yes", "YES"]
end
