defmodule CrucibleBumblebee.ModelLoader do
  @moduledoc """
  Loads configured V5 Bumblebee model profiles.
  """

  alias CrucibleBumblebee.{Backend, ModelBundle}
  alias CrucibleBumblebee.ModelLoader.{LoadError, Options}

  def default_model_id, do: Options.default_model_id()

  def infer_model_family(model_id) when is_binary(model_id), do: model_family(model_id, nil)

  def load!(opts \\ []) do
    case load(opts) do
      {:ok, bundle} -> bundle
      {:error, diagnostic} -> raise LoadError, diagnostic: diagnostic
    end
  end

  def load(opts \\ [])
  def load(%Options{} = options), do: do_load(options)

  def load(opts) do
    opts
    |> Options.from_env()
    |> do_load()
  end

  defp do_load(%Options{} = options) do
    with {:ok, selected_backend} <- Backend.prefer(options.backend),
         {:ok, module} <- model_module(options),
         {:ok, model_info} <- load_model(options, selected_backend, module),
         {:ok, tokenizer} <- load_tokenizer(options),
         {:ok, generation_config} <- load_generation_config(options, model_info.spec) do
      diagnostics = success_diagnostic(options, selected_backend, module, model_info)
      maybe_write_diagnostic!(options, diagnostics)

      {:ok,
       %ModelBundle{
         model_id: options.model_id,
         tokenizer_id: options.tokenizer_id,
         model: model_info.model,
         params: model_info.params,
         tokenizer: tokenizer,
         generation_config: generation_config,
         backend: selected_backend,
         model_family: model_family(options.model_id, model_info.spec),
         revision: options.revision,
         spec: model_info.spec,
         diagnostics: diagnostics
       }}
    else
      {:error, reason} ->
        diagnostic = diagnostic(options, reason)
        maybe_write_diagnostic!(options, diagnostic)
        {:error, diagnostic}

      error ->
        diagnostic = diagnostic(options, error)
        maybe_write_diagnostic!(options, diagnostic)
        {:error, diagnostic}
    end
  end

  def write_diagnostic!(path, diagnostic) when is_binary(path) and is_map(diagnostic) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(json_safe(diagnostic), pretty: true) <> "\n")
    path
  end

  defp load_model(%Options{} = options, backend, module) do
    opts =
      [
        module: module,
        architecture: options.architecture
      ] ++ model_backend_opts(backend)

    Bumblebee.load_model(Options.repository(options), opts)
  rescue
    error -> {:error, {:model_load_exception, Exception.message(error)}}
  end

  defp load_tokenizer(%Options{} = options) do
    Bumblebee.load_tokenizer(Options.repository(options, options.tokenizer_id))
  rescue
    error -> {:error, {:tokenizer_load_exception, Exception.message(error)}}
  end

  defp load_generation_config(%Options{} = options, spec) do
    Bumblebee.load_generation_config(Options.repository(options), spec_module: spec.__struct__)
  rescue
    _error -> {:ok, nil}
  end

  defp model_backend_opts(:binary), do: [backend: Nx.BinaryBackend]
  defp model_backend_opts(:exla_cpu), do: [backend: {EXLA.Backend, device: :cpu}]
  defp model_backend_opts(:exla_cuda), do: [backend: {EXLA.Backend, device: :cuda}]
  defp model_backend_opts(_backend), do: []

  defp model_module(%Options{module: module}) when is_atom(module) and not is_nil(module),
    do: {:ok, module}

  defp model_module(%Options{model_id: model_id}) do
    id = String.downcase(model_id)

    cond do
      String.contains?(id, "qwen3") -> {:ok, Bumblebee.Text.Qwen3}
      String.contains?(id, "gemma") -> {:ok, Bumblebee.Text.Gemma}
      String.contains?(id, "phi-3") or String.contains?(id, "phi3") -> {:ok, Bumblebee.Text.Phi3}
      String.contains?(id, "phi") -> {:ok, Bumblebee.Text.Phi}
      String.contains?(id, "roberta") -> {:ok, Bumblebee.Text.Roberta}
      String.contains?(id, "distilbert") -> {:ok, Bumblebee.Text.Distilbert}
      String.contains?(id, "bert") -> {:ok, Bumblebee.Text.Bert}
      String.contains?(id, "bart") -> {:ok, Bumblebee.Text.Bart}
      String.contains?(id, "gpt2") -> {:ok, Bumblebee.Text.Gpt2}
      true -> {:error, {:unsupported_model_family, model_id}}
    end
  end

  defp model_family(model_id, _spec) do
    id = String.downcase(model_id)

    cond do
      String.contains?(id, "qwen3") -> :qwen3
      String.contains?(id, "gemma") -> :gemma
      String.contains?(id, "phi") -> :phi
      String.contains?(id, "roberta") -> :roberta
      String.contains?(id, "distilbert") -> :distilbert
      String.contains?(id, "bert") -> :bert
      String.contains?(id, "bart") -> :bart
      String.contains?(id, "gpt2") -> :gpt2
      true -> :unknown
    end
  end

  defp success_diagnostic(%Options{} = options, backend, module, model_info) do
    %{
      ok: true,
      model_id: options.model_id,
      tokenizer_id: options.tokenizer_id,
      revision: options.revision,
      backend: backend,
      module: inspect(module),
      architecture: options.architecture,
      cache_dir: options.cache_dir || Bumblebee.cache_dir(),
      cache_status: Options.cache_status(options),
      offline?: options.offline?,
      spec_module: inspect(model_info.spec.__struct__)
    }
  end

  defp diagnostic(%Options{} = options, reason) do
    %{
      ok: false,
      model_id: options.model_id,
      tokenizer_id: options.tokenizer_id,
      revision: options.revision,
      requested_backend: options.backend,
      cache_dir: options.cache_dir || Bumblebee.cache_dir(),
      cache_status: Options.cache_status(options),
      offline?: options.offline?,
      reason: reason
    }
  end

  defp maybe_write_diagnostic!(%Options{diagnostic_path: nil}, _diagnostic), do: :ok

  defp maybe_write_diagnostic!(%Options{diagnostic_path: path}, diagnostic),
    do: write_diagnostic!(path, diagnostic)

  defp json_safe(value) when is_map(value) do
    Map.new(value, fn {key, value} -> {key, json_safe(value)} end)
  end

  defp json_safe(value) when is_tuple(value),
    do: value |> Tuple.to_list() |> Enum.map(&json_safe/1)

  defp json_safe(value) when is_list(value), do: Enum.map(value, &json_safe/1)
  defp json_safe(value), do: value
end
