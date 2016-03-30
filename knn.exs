defmodule KNN do
  def majority_vote(labels) do
    Enum.reduce(labels, Map.new, fn(l, m) -> Map.update(m, l, 1, &(&1+1)) end)
    |> Enum.max_by(fn {_, v} -> v end)            
  end  

  def classify(k, data_points, point, dist_fn) do
    top_k = Map.keys(data_points)
         |> Enum.sort_by(fn(curr_pt) -> dist_fn.(curr_pt, point) end)
         |> Enum.take(k)
    Enum.map(top_k, fn(pt) -> Map.get(data_points, pt) end) |>  majority_vote   
  end
end  

ExUnit.start

defmodule KNNTest do
  use ExUnit.Case

  test "KNN classify" do
    zipmult = fn(v1, v2) -> for {a, b} <- Enum.zip(v1, v2), do: a * b end
    dot = fn(v1, v2) -> Enum.reduce(zipmult.(v1, v2), 0, &+/2) end
    sum_of_sqs = fn(v) -> dot.(v, v) end 
    vect_diff = fn(v1, v2) -> for {a, b} <- Enum.zip(Tuple.to_list(v1), Tuple.to_list(v2)), do: a * b end 
    sq_dist = fn(loc1, loc2) -> sum_of_sqs.(vect_diff.(loc1, loc2)) end
    distance = fn(loc1, loc2) -> :math.sqrt(sq_dist.(loc1, loc2)) end
  
    data = %{{56, 2} => 'Happy', {3, 20} => 'Not Happy', {18, 1} => 'Happy', {20, 14} => 'Not Happy', {30, 30} => 'Happy', {35,  35}  => 'Happy'}
 
    assert KNN.classify(3, data, {10, 10}, distance)  == {'Not Happy', 2}
    #assert KNN.classify(3, data, {40, 40}, distance)  == 'Happy'
  end
end
