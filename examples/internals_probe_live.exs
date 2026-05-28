live? = CrucibleBumblebee.Config.live_enabled?()

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
      rungs: :string,
      artifact_root: :string,
      gate: :boolean
    ]
  )

unless live? do
  IO.inspect(%{
    ok: true,
    example: "internals_probe_live",
    skipped: true,
    reason: :live_not_enabled,
    run: "CRUCIBLE_BUMBLEBEE_LIVE=true mix run examples/internals_probe_live.exs"
  })

  if gate?, do: System.halt(1), else: System.halt(0)
end

backend =
  opts
  |> Keyword.get(:backend, CrucibleBumblebee.Config.backend("binary"))
  |> CrucibleBumblebee.ModelLoader.Options.normalize_backend()

models =
  CrucibleBumblebee.LiveMatrix.model_ladder()
  |> then(fn models ->
    case Keyword.get(opts, :rungs) do
      nil ->
        Enum.reject(models, &Map.has_key?(&1, :expected_blocker))

      rungs ->
        allowed = rungs |> String.split(",", trim: true) |> MapSet.new()
        Enum.filter(models, &MapSet.member?(allowed, &1.rung))
    end
  end)

results =
  Enum.map(models, fn model ->
    CrucibleBumblebee.InternalsProbe.run_model(model,
      backend: backend,
      artifact_root: Keyword.get(opts, :artifact_root)
    )
  end)

CrucibleBumblebee.MatrixReport.write_from_artifacts!(
  artifact_root: Keyword.get(opts, :artifact_root)
)

result = %{ok: Enum.all?(results, & &1.ok), results: results}
IO.inspect(result)

if gate? and not result.ok, do: System.halt(7)
