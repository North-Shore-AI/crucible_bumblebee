defmodule CrucibleBumblebee.CacheSummary do
  @moduledoc """
  Bounded metadata summaries for Bumblebee decoder caches.
  """

  def summarize(nil), do: %{}

  def summarize(cache) when is_map(cache) do
    %{
      keys: cache |> Map.keys() |> Enum.map(&to_string/1) |> Enum.sort(),
      blocks: cache |> Map.get(:blocks, Map.get(cache, "blocks")) |> block_count()
    }
  end

  def summarize(cache) when is_tuple(cache), do: %{tuple_size: tuple_size(cache)}
  def summarize(_cache), do: %{present?: true}

  defp block_count(nil), do: nil
  defp block_count(blocks) when is_tuple(blocks), do: tuple_size(blocks)
  defp block_count(blocks) when is_list(blocks), do: length(blocks)
  defp block_count(_blocks), do: nil
end
