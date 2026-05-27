defmodule CrucibleBumblebee.ModelLoader do
  @moduledoc """
  Loads the bounded V4 Bumblebee model profile.
  """

  alias CrucibleBumblebee.{Backend, ModelBundle}

  @default_model_id "hf-internal-testing/tiny-random-gpt2"

  def default_model_id, do: @default_model_id

  def load!(opts \\ []) do
    case load(opts) do
      {:ok, bundle} -> bundle
      {:error, reason} -> raise "model load failed: #{inspect(reason)}"
    end
  end

  def load(opts \\ []) do
    model_id =
      Keyword.get(
        opts,
        :model_id,
        System.get_env("CRUCIBLE_BUMBLEBEE_MODEL_ID") || @default_model_id
      )

    tokenizer_id =
      Keyword.get(
        opts,
        :tokenizer_id,
        System.get_env("CRUCIBLE_BUMBLEBEE_TOKENIZER_ID") || model_id
      )

    backend = Keyword.get(opts, :backend, backend_from_env())

    with {:ok, selected_backend} <- Backend.prefer(backend),
         {:ok, model_info} <- load_model(model_id, selected_backend),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer({:hf, tokenizer_id}),
         {:ok, generation_config} <- load_generation_config(model_id, model_info.spec) do
      {:ok,
       %ModelBundle{
         model_id: model_id,
         tokenizer_id: tokenizer_id,
         model: model_info.model,
         params: model_info.params,
         tokenizer: tokenizer,
         generation_config: generation_config,
         backend: selected_backend,
         model_family: model_family(model_id, model_info.spec),
         revision: Keyword.get(opts, :revision),
         spec: model_info.spec
       }}
    else
      {:error, reason} -> {:error, diagnostic(model_id, tokenizer_id, reason)}
      error -> {:error, diagnostic(model_id, tokenizer_id, error)}
    end
  end

  defp load_model(model_id, backend) do
    opts =
      [
        module: Bumblebee.Text.Gpt2,
        architecture: :for_causal_language_modeling
      ] ++ model_backend_opts(backend)

    Bumblebee.load_model({:hf, model_id}, opts)
  rescue
    error -> {:error, {:model_load_exception, Exception.message(error)}}
  end

  defp load_generation_config(model_id, spec) do
    Bumblebee.load_generation_config({:hf, model_id}, spec_module: spec.__struct__)
  rescue
    _error -> {:ok, nil}
  end

  defp model_backend_opts(:binary), do: [backend: Nx.BinaryBackend]
  defp model_backend_opts(:exla_cpu), do: [backend: {EXLA.Backend, device: :cpu}]
  defp model_backend_opts(:exla_cuda), do: [backend: {EXLA.Backend, device: :cuda}]
  defp model_backend_opts(_backend), do: []

  defp backend_from_env do
    (System.get_env("CRUCIBLE_BUMBLEBEE_BACKEND") || System.get_env("CRUCIBLE_BACKEND") ||
       "auto")
    |> String.to_atom()
  end

  defp model_family(model_id, _spec) do
    cond do
      String.contains?(String.downcase(model_id), "gpt2") -> :gpt2
      true -> :unknown
    end
  end

  defp diagnostic(model_id, tokenizer_id, reason) do
    %{
      model_id: model_id,
      tokenizer_id: tokenizer_id,
      cache_dir:
        System.get_env("CRUCIBLE_MODEL_CACHE_DIR") || Path.expand("~/.cache/huggingface/hub"),
      reason: reason
    }
  end
end
