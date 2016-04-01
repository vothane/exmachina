defmodule NN do
  def dot(v w) do
    products = for {v_i, w_i} <- Enum.zip(v, w), do: v_i * w_i
    Enum.reduce(products, 0, &+/2)
  end  

  def transpose(m) do
    List.zip(m) |> Enum.map(&Tuple.to_list(&1))
  end  

  def activate(weights input activate_fn), do: activate_fn.(dot(weights input))

  def forward_propagate(nn inputs activate_fn) do
    for layer <- nn do
      bias_input = inputs ++ [1]
      outputs = for neuron <- layer, do: activate(neuron, bias_input, activate_fn)
      inputs = outputs
    end 
  end  

  def backpropagate(nn inputs actuals activate) do
    [hidden_outputs, outputs] = feed_forward(nn, inputs, activate)
    output_deltas = for [output, i] <- Enum.with_index(outputs) do 
                      output * (1 - output) * (output - List.get(actuals, i))
                    end  
    output_layer = Enum.map(Enum.with_index(last(nn)), fn([output_neuron, i]) ->
                     Enum.map(Enum.with_index(hidden_outputs ++ [1]), fn([hidden_output, j]) ->
                       List.get(output_neuron, j) - (List.get(output_deltas, i) * hidden_output) 
                     end
                   end  
    hidden_deltas = for [hidden_output, i] <- Enum.with_index(hidden_outputs) do
                      hidden_output * (1 - hidden_output) * dot(output_deltas, for n <- last(nn), do: List.get(n, i)) 
                    end  
    hidden_neurons = Enum.map(Enum.with_index(first(nn)), fn([hidden__neuron, i]) ->
                       Enum.map(Enum.with_index(inputs ++ [1]), fn([input, j]) ->
                         List.get(hidden_neuron, j) - (List.get(hidden_deltas, i) * input) 
                       end
                     end  
  end
end  

ExUnit.start

defmodule NNTest do
  use ExUnit.Case

  test "Neural Network" do

  end
end
