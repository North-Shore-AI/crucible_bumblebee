defmodule CrucibleBumblebee.ActivationMapperTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.ActivationMapper

  test "maps Bumblebee outputs to canonical activation metadata" do
    assert ActivationMapper.final_logits().activation_name == "unembed.hook_logits"

    assert %{
             activation_name: "blocks.0.hook_resid_pre",
             axes: [:batch, :pos, :d_model],
             capture_exactness: :bumblebee_output_hidden_state
           } = ActivationMapper.hidden_state(0, 3)

    assert %{
             activation_name: "blocks.1.hook_resid_post",
             layer_index: 1
           } = ActivationMapper.hidden_state(2, 3)

    assert %{
             activation_name: "blocks.4.attn.hook_pattern",
             axes: [:batch, :head, :dest_pos, :src_pos]
           } = ActivationMapper.attention_weights(4)

    assert %{
             activation_name: "blocks.4.attn.hook_attn_scores",
             axes: [:batch, :head, :dest_pos, :src_pos]
           } = ActivationMapper.attention_scores(4)
  end

  test "maps surface-declared deep nodes to canonical activation names" do
    assert ActivationMapper.surface_metadata(:attention_q, 2).activation_name ==
             "blocks.2.attn.hook_q"

    assert ActivationMapper.surface_metadata(:mlp_gates, 2).activation_name ==
             "blocks.2.mlp.hook_pre"
  end
end
