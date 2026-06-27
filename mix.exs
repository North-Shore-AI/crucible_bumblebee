defmodule CrucibleBumblebee.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/crucible_bumblebee"
  @bumblebee_ref "068fb2958672706dfc2f8c2b2d9b2c88bffc540a"

  def project do
    [
      app: :crucible_bumblebee,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
      name: "CrucibleBumblebee",
      description: "Bumblebee, Axon, and Nx adapter layer for Crucible tap plans and traces",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs(),
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  def cli do
    [
      preferred_envs: [
        ci: :test,
        docs: :dev
      ]
    ]
  end

  defp deps do
    [
      {:crucible_signal, path: "../crucible_signal"},
      {:crucible_tap, path: "../crucible_tap"},
      {:crucible_signal_trace, path: "../crucible_signal_trace"},
      {:crucible_mechinterp, path: "../crucible_mechinterp"},
      {:crucible_policy, path: "../crucible_policy"},
      {:nx, "~> 0.12", override: true},
      {:axon, "~> 0.7"},
      {:bumblebee, github: "North-Shore-AI/bumblebee", ref: @bumblebee_ref, override: true},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.40.1", only: [:dev, :test], runtime: false}
    ]
  end

  defp aliases do
    [
      ci: [
        "deps.get",
        "format --check-formatted",
        "compile --warnings-as-errors",
        "test",
        "docs"
      ]
    ]
  end

  defp docs do
    [
      main: "readme",
      extras:
        ["README.md", "CHANGELOG.md", "LICENSE"] ++
          Path.wildcard("guides/*.md") ++ Path.wildcard("docs/*.md"),
      groups_for_extras: [
        Guides: Path.wildcard("guides/*.md"),
        "API Notes": Path.wildcard("docs/*.md")
      ],
      source_ref: "main",
      source_url: @source_url,
      homepage_url: @source_url
    ]
  end

  defp package do
    [
      name: "crucible_bumblebee",
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib assets guides docs examples priv mix.exs README.md LICENSE CHANGELOG.md)
    ]
  end
end
