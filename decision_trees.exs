defmodule DecisionTrees do

  def entropy(class_probs) do
    Enum.map(class_probs, fn(p) -> -p * :math.log2(p) end)
    |> Enum.sum
  end 
 
  def class_probs(labels) do
    total = Enum.count(labels)
    Enum.reduce(labels, Map.new, fn(l, m) -> Map.update(m, l, 1, &(&1 + 1)) end) 
    |> Enum.map(fn {_, count} -> count / total end) 
  end  
end  

ExUnit.start

defmodule DecisionTreesTest do
  use ExUnit.Case

  test "calculate entropy from class probabilities" do
    assert DecisionTrees.entropy([0.6, 0.4]) == 0.9709505944546686
  end

  test "calculate class probabilities from labels" do
    assert DecisionTrees.class_probs([false, false, false, true, true]) == [0.6, 0.4]
  end  
end

