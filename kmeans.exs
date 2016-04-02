defmodule KMeans do

  def distance(loc1, loc2) do
    zipmult = fn(v1, v2) -> for {a, b} <- Enum.zip(v1, v2), do: a * b end
    dot = fn(v1, v2) -> Enum.reduce(zipmult.(v1, v2), 0, &+/2) end
    sum_of_sqs = fn(v) -> dot.(v, v) end 
    vect_diff = fn(v1, v2) -> for {a, b} <- Enum.zip(v1, v2), do: a * b end 
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
    Enum.min(for mean <- means, do: distance(input, mean))
  end    
                         
  def train(inputs, k) do
    means = for _ <- 1..k, do: Enum.random(inputs)
    assign(k, inputs, means, nil, false)
  end
  
  defp assign(k, inputs, means, prev_assignments, stop) when stop == false do 
    curr_assignments = for input <- inputs, do: classify(input, means)

    for i <- 1..k do
      i_pts = for {p, a} <- Enum.zip(inputs, curr_assignments), do: if a == i, do: p
      unless Enum.any?(i_pts, &is_nil/1), do: means = List.replace_at(means, i, vector_mean(i_pts))  
    end
    
    assign(k, inputs, means, curr_assignments, prev_assignments == curr_assignments)
  end   
  
  defp assign(k, inputs, means, prev_assignments, stop) when stop == true, do: prev_assignments
end  

ExUnit.start

defmodule KMeansTest do
  use ExUnit.Case

  test "KMeans classify" do
  
    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],
              [26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],
              [-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    
    k=3
    kmeans = KMeans.train(inputs, 3)
    IO.inspect kmeans
    #assert KMeans.classify([0,0], kmeans)  == 
  end
end
