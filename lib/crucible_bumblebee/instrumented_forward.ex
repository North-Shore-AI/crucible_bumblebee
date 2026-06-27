defmodule CrucibleBumblebee.InstrumentedForward do
  @moduledoc """
  Validates that a compiled Bumblebee forward run returned the requested outputs.

  The runner cannot retrofit global Bumblebee options onto an arbitrary
  `predict_fun`, so it fails closed when a compiled tap plan requires output keys
  that are absent from the returned map.
  """

  alias CrucibleBumblebee.CompiledTapPlan

  @doc "Validates output keys implied by the compiled tap plan."
  @spec validate_outputs(map(), CompiledTapPlan.t()) :: :ok | {:error, term()}
  def validate_outputs(outputs, %CompiledTapPlan{} = compiled_taps) when is_map(outputs) do
    compiled_taps.global_layer_options
    |> Enum.reduce_while(:ok, fn
      {:output_hidden_states, true}, :ok ->
        require_output(outputs, :hidden_states)

      {:output_attentions, true}, :ok ->
        require_output(outputs, :attentions)

      {_option, _value}, :ok ->
        {:cont, :ok}
    end)
  end

  def validate_outputs(outputs, _compiled_taps),
    do: {:error, {:invalid_forward_outputs, outputs}}

  defp require_output(outputs, key) do
    if Map.has_key?(outputs, key) or Map.has_key?(outputs, Atom.to_string(key)) do
      {:cont, :ok}
    else
      {:halt, {:error, {:missing_compiled_output, key}}}
    end
  end
end
