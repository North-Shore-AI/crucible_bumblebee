defmodule CrucibleBumblebee.LogitsProcessor do
  @moduledoc """
  Adapter for applying Crucible steering plans to logits.
  """

  alias CruciblePolicy.{LogitSteering, SteeringPlan}

  def process(logits, %SteeringPlan{} = steering_plan),
    do: LogitSteering.apply(logits, steering_plan)
end
