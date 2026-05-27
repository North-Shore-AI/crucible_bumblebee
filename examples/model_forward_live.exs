if System.get_env("CRUCIBLE_BUMBLEBEE_LIVE") in ["1", "true"] do
  IO.inspect(%{ok: false, example: "model_forward_live", reason: :live_model_not_configured})
else
  IO.inspect(%{ok: true, example: "model_forward_live", skipped: true, reason: :live_not_enabled})
end
