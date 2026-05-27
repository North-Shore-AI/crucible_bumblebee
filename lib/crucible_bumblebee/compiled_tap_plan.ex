defmodule CrucibleBumblebee.CompiledTapPlan do
  @moduledoc """
  Bumblebee-specific compiled tap plan.
  """

  @derive Jason.Encoder
  defstruct tap_plan_id: nil,
            global_layer_options: [],
            hook_names: [],
            matched: [],
            unsupported_optional: [],
            report: nil,
            metadata: %{}

  @type t :: %__MODULE__{}
end
