defmodule CrucibleBumblebee.ModelBundle do
  @moduledoc """
  Loaded Bumblebee model, tokenizer, params, and execution metadata.
  """

  @derive {Inspect, except: [:model, :params, :tokenizer]}
  defstruct [
    :model_id,
    :tokenizer_id,
    :model,
    :params,
    :tokenizer,
    :generation_config,
    :backend,
    :model_family,
    :revision,
    :spec
  ]
end
