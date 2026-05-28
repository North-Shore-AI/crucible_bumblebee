defmodule CrucibleBumblebee.ModelLoader.Options do
  @moduledoc "Resolved model-loader options."

  alias CrucibleBumblebee.Config

  @default_model_id "hf-internal-testing/tiny-random-gpt2"
  @default_prompt "Hi"
  @backend_values %{
    "auto" => :auto,
    "binary" => :binary,
    "exla_cpu" => :exla_cpu,
    "exla-cpu" => :exla_cpu,
    "exla_cuda" => :exla_cuda,
    "exla-cuda" => :exla_cuda,
    "torchx_cpu" => :torchx_cpu,
    "torchx-cpu" => :torchx_cpu,
    "torchx_cuda" => :torchx_cuda,
    "torchx-cuda" => :torchx_cuda
  }

  @derive Jason.Encoder
  defstruct model_id: @default_model_id,
            tokenizer_id: nil,
            revision: nil,
            backend: :auto,
            offline?: false,
            cache_dir: nil,
            prompt: @default_prompt,
            max_new_tokens: 8,
            seed: nil,
            artifact_root: nil,
            architecture: :for_causal_language_modeling,
            module: nil,
            diagnostic_path: nil

  @type t :: %__MODULE__{}

  def default_model_id, do: @default_model_id

  def from_env(overrides \\ [], env_reader \\ &Config.env/1) do
    new(Config.model_loader_attrs(overrides, env_reader))
  end

  def new(attrs \\ []) when is_list(attrs) or is_map(attrs) do
    attrs = normalize_attrs(attrs)
    model_id = Map.get(attrs, :model_id, @default_model_id)

    %__MODULE__{
      model_id: model_id,
      tokenizer_id: Map.get(attrs, :tokenizer_id) || model_id,
      revision: Map.get(attrs, :revision),
      backend: normalize_backend(Map.get(attrs, :backend, :auto)),
      offline?: Map.get(attrs, :offline?, false) in [true, "true", "1", 1],
      cache_dir: Map.get(attrs, :cache_dir),
      prompt: Map.get(attrs, :prompt, @default_prompt),
      max_new_tokens: Map.get(attrs, :max_new_tokens, 8),
      seed: Map.get(attrs, :seed),
      artifact_root: Map.get(attrs, :artifact_root),
      architecture: Map.get(attrs, :architecture, :for_causal_language_modeling),
      module: Map.get(attrs, :module),
      diagnostic_path: Map.get(attrs, :diagnostic_path)
    }
  end

  def repository(%__MODULE__{} = options, id \\ nil) do
    repo_opts =
      []
      |> maybe_put(:revision, options.revision)
      |> maybe_put(:cache_dir, options.cache_dir)
      |> maybe_put(:offline, options.offline?)

    {:hf, id || options.model_id, repo_opts}
  end

  def cache_status(%__MODULE__{cache_dir: nil}), do: :default_cache

  def cache_status(%__MODULE__{cache_dir: cache_dir}) do
    if File.dir?(cache_dir), do: :cache_dir_present, else: :cache_dir_missing
  end

  def normalize_backend(backend) when is_atom(backend), do: backend

  def normalize_backend(backend) when is_binary(backend) do
    Map.get(@backend_values, String.downcase(backend), :unknown)
  end

  def normalize_backend(_backend), do: :unknown

  defp maybe_put(keyword, _key, nil), do: keyword
  defp maybe_put(keyword, _key, false), do: keyword
  defp maybe_put(keyword, key, value), do: keyword ++ [{key, value}]

  defp normalize_attrs(attrs) when is_list(attrs), do: attrs |> Map.new() |> normalize_attrs()

  defp normalize_attrs(attrs) when is_map(attrs) do
    Map.new(attrs, fn
      {key, value} when is_binary(key) -> {String.to_existing_atom(key), value}
      {key, value} -> {key, value}
    end)
  rescue
    ArgumentError -> Map.new(attrs)
  end
end
