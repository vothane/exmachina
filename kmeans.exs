defmodule KMeans do

  def distance(loc1, loc2) do
    vect_mult = fn(v1, v2) -> for {a, b} <- Enum.zip(v1, v2), do: a * b end
    dot = fn(v1, v2) -> Enum.reduce(vect_mult.(v1, v2), 0, &+/2) end
    sum_of_sqs = fn(v) -> dot.(v, v) end 
    vect_diff = fn(v1, v2) -> for {a, b} <- Enum.zip(v1, v2), do: a - b end 
    sq_dist = fn(loc1, loc2) -> sum_of_sqs.(vect_diff.(loc1, loc2)) end
    
    :math.sqrt(sq_dist.(loc1, loc2))
  end 
  
  def vector_mean(vectors) do
    vector_add = fn(v, w) -> for {v_i, w_i} <- Enum.zip(v,w), do: v_i + w_i end
    vector_sum = fn(vectors) -> Enum.reduce(vectors, vector_add) end
    scalar_multiply = fn(c, v) -> for v_i <- v, do: c * v_i end
 
    n = Enum.count(vectors)
    scalar_multiply.(1/n, vector_sum.(vectors))
  end 
  
  def classify(input, means) do 
    dists = for mean <- means, do: distance(input, mean)
    min = Enum.min(dists)
    Enum.find_index(dists, fn(dist) -> dist == min end)
  end    
                         
  def train(inputs, k) do
    means = for _ <- 1..k, do: Enum.random(inputs)
    assign(k, inputs, means, nil, false)
  end
  
  defp assign(k, inputs, means, prev_assignments, stop) when stop == false do 
    curr_assignments = for input <- inputs, do: classify(input, means)
    grouping = Enum.zip(inputs, curr_assignments)
            |> Enum.group_by(fn {_, assign_id} -> assign_id end)
    extract = fn(vecs) -> Enum.map(vecs, fn {vec, i} -> vec end) end
    means = Enum.reduce(grouping, means, fn({i,vecs}, means) -> 
              List.replace_at(means, i, vector_mean(extract.(vecs))) end)  
        
    assign(k, inputs, means, curr_assignments, prev_assignments == curr_assignments)
  end   
  
  defp assign(k, inputs, means, prev_assignments, stop) when stop == true, do: means
end  

ExUnit.start

defmodule KMeansTest do
  use ExUnit.Case

  test "KMeans classify" do
    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],
              [26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],
              [-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

    k3_means = [[-43.800000000000004, 5.4], 
                [-15.888888888888888, -10.333333333333332], 
                [18.333333333333332, 19.833333333333332]]

    k2_means = [[-25.857142857142854, -4.714285714285714], 
                [18.333333333333332, 19.833333333333332]]

    k = 3
    kmeans = KMeans.train(inputs, k)
    IO.inspect kmeans

    assert Enum.all?(kmeans, fn(mean) -> Enum.member?(k3_means, mean) end)

    assignment_0_0 = Enum.find_index(kmeans, fn el -> el == [-15.888888888888888, -10.333333333333332] end)
    assert KMeans.classify([0,0], kmeans)  == assignment_0_0

    assignment_10_10 = Enum.find_index(kmeans, fn el -> el == [18.333333333333332, 19.833333333333332] end)
    assert KMeans.classify([10,10], kmeans)  == assignment_10_10

    k = 2
    kmeans = KMeans.train(inputs, k)
    IO.inspect kmeans

    assert Enum.all?(kmeans, fn(mean) -> Enum.member?(k2_means, mean) end)

    assignment_0_0 = Enum.find_index(kmeans, fn el -> el == [-25.857142857142854, -4.714285714285714] end)
    assert KMeans.classify([0,0], kmeans)  == assignment_0_0

    assignment_10_10 = Enum.find_index(kmeans, fn el -> el == [18.333333333333332, 19.833333333333332] end)
    assert KMeans.classify([10,10], kmeans)  == assignment_10_10
  end
end

