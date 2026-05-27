defmodule CrucibleBumblebee.ModelSurfaceTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{ExampleSurface, ModelSurface, Qwen3Surface, SurfacePreflight}

  test "example surface conforms to model surface behaviour" do
    surface = ExampleSurface.surface(num_blocks: 1)

    assert %ModelSurface{id: :example_transformer, family: :example_transformer} = surface
    assert surface.capabilities.logits_processors

    assert [output_hidden_states: true, output_attentions: true] =
             ModelSurface.output_options(surface, %{})
  end

  test "preflight writes a versioned artifact with dependency fingerprint" do
    root =
      Path.join(
        System.tmp_dir!(),
        "crucible-bumblebee-preflight-#{System.unique_integer([:positive])}"
      )

    surface = ExampleSurface.surface(num_blocks: 1)

    assert {:ok, artifact} = SurfacePreflight.run(surface, %{}, root: root)
    path = SurfacePreflight.artifact_path(surface.id, artifact.dependency_fingerprint, root: root)

    assert File.exists?(path)
    assert artifact.schema == "crucible_bumblebee.surface_preflight"
    assert "lm_head.output" in artifact.nodes
  end

  test "generic logit lens access delegates to the surface module" do
    params = %{decoder: %{final_norm: :norm}, lm_head: %{kernel: :kernel}}

    assert {:ok, access} = ModelSurface.logit_lens_access(ExampleSurface.surface(), %{}, params)
    assert access.final_norm == :norm
    assert access.unembedding == :kernel
  end

  test "unsupported logit lens access negotiates cleanly" do
    assert {:error, :unsupported} =
             ModelSurface.logit_lens_access(Qwen3Surface.surface(num_blocks: 1), %{}, %{})
  end
end
