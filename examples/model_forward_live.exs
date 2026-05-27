live? = System.get_env("CRUCIBLE_BUMBLEBEE_LIVE") in ["1", "true"]
gate? = "--gate" in System.argv()

unless live? do
  IO.inspect(%{
    ok: true,
    example: "model_forward_live",
    skipped: true,
    reason: :live_not_enabled,
    run: "CRUCIBLE_BUMBLEBEE_LIVE=true mix run examples/model_forward_live.exs"
  })

  if gate?, do: System.halt(1), else: System.halt(0)
end

try do
  result = CrucibleBumblebee.Live.forward()
  IO.inspect(result)
rescue
  error ->
    IO.puts(:stderr, Jason.encode!(%{ok: false, example: "model_forward_live", error: Exception.message(error)}))
    System.halt(6)
end
