defmodule DecisionTrees do

  def entropy(class_probs) do
    Enum.map(class_probs, fn(p) -> -p * :math.log2(p) end)
    |> Enum.sum
  end 
  
end  

ExUnit.start

defmodule DecisionTreesTest do
  use ExUnit.Case

  test "calculate entropy from class probabilities" do
    assert DecisionTrees.entropy([0.6, 0.4]) == 0.9709505944546686
  end
end

