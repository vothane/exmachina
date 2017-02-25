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

  def data_entropy(data) do
    Enum.map(data, fn {_, label} -> label end)
    |> DecisionTrees.class_probs
    |> DecisionTrees.entropy
  end  
end  

ExUnit.start

defmodule DecisionTreesTest do
  use ExUnit.Case

  setup_all do
    {:ok, data: [{%{'level'=>'Senior','lang'=>'Java','tweets'=>'no','phd'=>'no'},   false},
                 {%{'level'=>'Senior','lang'=>'Java','tweets'=>'no','phd'=>'yes'},  false},
                 {%{'level'=>'Mid','lang'=>'Python','tweets'=>'no','phd'=>'no'},     true},
                 {%{'level'=>'Junior','lang'=>'Python','tweets'=>'no','phd'=>'no'},  true},
                 {%{'level'=>'Junior','lang'=>'R','tweets'=>'yes','phd'=>'no'},      true},
                 {%{'level'=>'Junior','lang'=>'R','tweets'=>'yes','phd'=>'yes'},    false},
                 {%{'level'=>'Mid','lang'=>'R','tweets'=>'yes','phd'=>'yes'},        true},
                 {%{'level'=>'Senior','lang'=>'Python','tweets'=>'no','phd'=>'no'}, false},
                 {%{'level'=>'Senior','lang'=>'R','tweets'=>'yes','phd'=>'no'},      true},
                 {%{'level'=>'Junior','lang'=>'Python','tweets'=>'yes','phd'=>'no'}, true},
                 {%{'level'=>'Senior','lang'=>'Python','tweets'=>'yes','phd'=>'yes'},true},
                 {%{'level'=>'Mid','lang'=>'Python','tweets'=>'no','phd'=>'yes'},    true},
                 {%{'level'=>'Mid','lang'=>'Java','tweets'=>'yes','phd'=>'no'},      true},
                 {%{'level'=>'Junior','lang'=>'Python','tweets'=>'no','phd'=>'yes'},false}]}
  end

  test "calculate entropy from class probabilities" do
    assert DecisionTrees.entropy([0.6, 0.4]) == 0.9709505944546686
  end

  test "calculate class probabilities from labels" do
    assert DecisionTrees.class_probs([false, false, false, true, true]) == [0.6, 0.4]
  end  

  test "calculate entropy from labeled data", state do
    data  = state[:data]
    seniors = Enum.filter(data, fn({%{'level' => level}, _}) -> level == 'Senior' end)
    assert DecisionTrees.data_entropy(seniors) == 0.9709505944546686    
  end
end

