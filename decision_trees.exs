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

  def partition_entropy(subsets) do
    total_count = Enum.map(subsets, &Enum.count/1) |> Enum.sum              
    Enum.map(subsets, fn(subset) -> data_entropy(subset) * Enum.count(subset) / total_count end)
    |> Enum.sum
  end

  def partition_entropy_by(data, attr) do
    Enum.group_by(data, fn {%{^attr => v}, _} -> v end) 
    |> Map.values
    |> DecisionTrees.partition_entropy
  end

  def split_tree(data, split_attrs) do
    best_attr = Enum.min_by(split_attrs, &(partition_entropy_by(data, &1))) 
    partitions = Enum.group_by(data, fn {%{^best_attr => v}, _} -> v end)
    attrs = List.delete(split_attrs, best_attr)
    {best_attr, attrs, partitions}
  end

  def build_tree(data, attrs \\ nil) do
    split_attrs = case attrs do
                    nil -> {x, _} = List.first(data)
                           Map.keys(x)
                    attrs -> attrs       
                  end  
    num_inputs = Enum.count(data)
    num_trues = Enum.filter(data, fn {_, bool} -> bool == true end) |> Enum.count
    num_falses = num_inputs - num_trues

    cond do
      num_trues == 0 -> false
      num_falses == 0 -> true
      Enum.empty?(split_attrs) -> num_trues >= num_falses
      true ->
        {best_attr, subset_attrs, partitions} = split_tree(data, split_attrs) 
        subtrees = Enum.reduce(partitions, Map.new, fn({attr, subset}, map) -> Map.put(map, attr, DecisionTrees.build_tree(subset, subset_attrs)) end)
        {best_attr, subtrees}
    end 
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

  test "calculate entropy for split labeled data", state do
    data  = state[:data]
    seniors = Enum.filter(data, fn({%{'level' => level}, _}) -> level == 'Senior' end)

    assert DecisionTrees.data_entropy(seniors) == 0.9709505944546686    
  end

  test "calculate entropy for split subsets", state do
    data  = state[:data]
    seniors = Enum.filter(data, fn({%{'level' => level}, _}) -> level == 'Senior' end)
    mids = Enum.filter(data, fn({%{'level' => level}, _}) -> level == 'Mid' end)
    juniors = Enum.filter(data, fn({%{'level' => level}, _}) -> level == 'Junior' end)

    assert DecisionTrees.partition_entropy([seniors, mids, juniors]) == 0.6935361388961919    
  end

  test "building trees", state do
    data  = state[:data]

    assert DecisionTrees.build_tree(data) == {'level',
                                               %{'Junior' => {'phd', %{'no' => true, 'yes' => false}},
                                                 'Mid' => true,
                                                 'Senior' => {'tweets', %{'no' => false, 'yes' => true}}}}  
  end
end

