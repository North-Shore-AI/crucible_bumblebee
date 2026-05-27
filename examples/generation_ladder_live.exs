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
      backend: :string,
      limit: :integer,
      rungs: :string,
      max_new_tokens: :integer,
      high_level: :boolean,
      artifact_root: :string,
      gate: :boolean
    ]
  )

unless live? do
  IO.inspect(%{
    ok: true,
    example: "generation_ladder_live",
    skipped: true,
    reason: :live_not_enabled,
    run: "CRUCIBLE_BUMBLEBEE_LIVE=true mix run examples/generation_ladder_live.exs"
  })

  if gate?, do: System.halt(1), else: System.halt(0)
end

backend =
  opts
  |> Keyword.get(:backend, System.get_env("CRUCIBLE_BUMBLEBEE_BACKEND") || "binary")
  |> CrucibleBumblebee.ModelLoader.Options.normalize_backend()

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
  |> then(fn models ->
    case Keyword.get(opts, :limit) do
      nil -> models
      limit -> Enum.take(models, limit)
    end
  end)

result =
  CrucibleBumblebee.LiveMatrix.run_generation_ladder(
    backend: backend,
    models: models,
    max_new_tokens: Keyword.get(opts, :max_new_tokens, 1),
    attempt_high_level_generation?: Keyword.get(opts, :high_level, false),
    artifact_root: Keyword.get(opts, :artifact_root)
  )

IO.inspect(result)

if gate? and not result.ok, do: System.halt(7)
