defmodule NN do
  def dot(v, w) do
    products = for {v_i, w_i} <- Enum.zip(v, w), do: v_i * w_i
    Enum.reduce(products, 0, &+/2)
  end  

  def transpose(m) do
    List.zip(m) |> Enum.map(&Tuple.to_list(&1))
  end  

  def sigmoid(x), do: 1 / (1 + :math.exp(-x))

  def activate(weights, input), do: sigmoid(dot(weights, input))

  def forward_propagate(nn, inputs) do
    for layer <- nn do
      bias_input = inputs ++ [1]
      IO.inspect layer
      outputs = for neuron <- layer, do: activate(neuron, bias_input)
      inputs = outputs
    end 
  end  

  def backward_propagate(nn, inputs, actuals) do
    [hidden_outputs, outputs] = forward_propagate(nn, inputs)

    output_deltas = for {output, i} <- Enum.with_index(outputs) do 
                      output * (1 - output) * (output - Enum.at(actuals, i))
                    end  

    output_layer = Enum.map(Enum.with_index(List.last(nn)), fn {output_neuron, i} ->
                     Enum.map(Enum.with_index(hidden_outputs ++ [1]), fn {hidden_output, j} ->
                       Enum.at(output_neuron, j) - (Enum.at(output_deltas, i) * hidden_output) 
                     end)
                   end)

    hidden_deltas = for {hidden_output, i} <- Enum.with_index(hidden_outputs) do
                      hidden_output * (1 - hidden_output) * dot(output_deltas, (for n <- List.last(nn), do: Enum.at(n, i))) 
                    end  

    hidden_layer = Enum.map(Enum.with_index(List.first(nn)), fn {hidden_neuron, i} ->
                       Enum.map(Enum.with_index(inputs ++ [1]), fn {input, j} ->
                       Enum.at(hidden_neuron, j) - (Enum.at(hidden_deltas, i) * input) 
                     end)
                   end) 
    [hidden_layer | output_layer]               
  end
end  

ExUnit.start

defmodule NNTest do
  use ExUnit.Case

  test "Neural Network" do
    training_set = [{[0, 0], [0]}, {[0, 1], [1]}, {[1, 0], [1]}, {[1, 1], [0]}]

    nn = [# hidden layer
          [[2, 2, -3],  # 'and' neuron
           [2, 2, -1]], # 'or' neuron
          # output layer
          [[-6, 6, -3]]]

    nn = Enum.reduce(1..5000, nn, fn(_, nn) -> 
           Enum.reduce(training_set, nn, fn({x, y}, nn) ->
             NN.backward_propagate(nn, x, y)
           end)
         end)
    IO.inspect(nn)     
  end
end
