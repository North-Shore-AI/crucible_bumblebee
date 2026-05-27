live? = System.get_env("CRUCIBLE_BUMBLEBEE_LIVE") in ["1", "true"]

argv =
  case System.argv() do
    ["--" | rest] -> rest
    args -> args
  end

gate? = "--gate" in argv

{opts, _rest, _invalid} =
  OptionParser.parse(argv,
    strict: [
      backends: :string,
      rungs: :string,
      artifact_root: :string,
      gate: :boolean
    ]
  )

unless live? do
  IO.inspect(%{
    ok: true,
    example: "backend_ladder_live",
    skipped: true,
    reason: :live_not_enabled,
    run: "CRUCIBLE_BUMBLEBEE_LIVE=true mix run examples/backend_ladder_live.exs"
  })

  if gate?, do: System.halt(1), else: System.halt(0)
end

models =
  CrucibleBumblebee.LiveMatrix.model_ladder()
  |> then(fn models ->
    case Keyword.get(opts, :rungs) do
      nil ->
        models

      rungs ->
        allowed = rungs |> String.split(",", trim: true) |> MapSet.new()
        Enum.filter(models, &MapSet.member?(allowed, &1.rung))
    end
  end)

backends =
  case Keyword.get(opts, :backends) do
    nil ->
      CrucibleBumblebee.LiveMatrix.backend_ladder()

    backends ->
      backends
      |> String.split(",", trim: true)
      |> Enum.map(&CrucibleBumblebee.ModelLoader.Options.normalize_backend/1)
  end

result =
  CrucibleBumblebee.LiveMatrix.run_backend_ladder(
    models: models,
    backends: backends,
    artifact_root: Keyword.get(opts, :artifact_root)
  )

IO.inspect(result)

if gate? and not result.ok, do: System.halt(7)
