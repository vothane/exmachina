defmodule NaiveBayes do
  def tokenize(text) do
    text
    |> String.downcase
    |> String.split
    |> Enum.uniq
    |> Enum.map(&String.to_atom/1)
  end  
  
  def count_words(counts, {message, is_spam}) do
    v = case is_spam do
          true -> [1, 0]
          false -> [0, 1]
        end  
    message_counts = Enum.reduce(tokenize(message), Keyword.new, fn(w, kw) -> Keyword.put(kw, w, v) end)
    Keyword.merge(counts, message_counts, fn(_, v1, v2) -> for {a, b} <- Enum.zip(v1, v2), do: a + b end)
  end

  def count(training_data) do
    Enum.reduce(training_data, Keyword.new, fn(m, kw) -> count_words(kw, m) end)
  end

  def word_probs(counts, total_spam, total_ham, k) do
    for {word, [num_spam, num_ham]} <- counts do
      {word, (num_spam + k) / (total_spam + 2 * k), (num_ham + k) / (total_ham + 2 * k)}
    end  
  end

  def spam_prob(word_probs, text) do
    text_words = tokenize(text)
    probs = %{:log_prob_if_spam => 0.0, :log_prob_if_not_spam => 0.0}

    probs = Enum.reduce(word_probs, probs, 
      fn({word, prob_if_spam, prob_if_not_spam}, probs) ->  
        if Enum.member?(text_words, word) do
          probs = Map.update!(probs, :log_prob_if_spam, &(&1 + :math.log(prob_if_spam)))
          Map.update!(probs, :log_prob_if_not_spam, &(&1 + :math.log(prob_if_not_spam)))
        else
          probs = Map.update!(probs, :log_prob_if_spam, &(&1 + :math.log(1.0 - prob_if_spam)))
          Map.update!(probs, :log_prob_if_not_spam, &(&1 + :math.log(1.0 - prob_if_not_spam)))
        end
      end)                  
    IO.inspect(probs)      
    prob_if_spam = :math.exp(Map.get(probs, :log_prob_if_spam))
    prob_if_not_spam = :math.exp(Map.get(probs, :log_prob_if_not_spam))
    prob_if_spam / (prob_if_spam + prob_if_not_spam)
  end  

  def train(training_data, k \\ 0.5) do
    num_spams = Enum.reduce(training_data, 0, fn({_, is_spam}, c) -> 
                                                if is_spam, do: c + 1, else: c end)
    num_non_spams = Enum.count(training_data) - num_spams
    word_counts = count(training_data)
    word_probs(word_counts, num_spams, num_non_spams, k)
  end                  
end  

ExUnit.start

defmodule NaiveBayesTest do
  use ExUnit.Case

  test "Naive Bayes classify" do
    training_data = [{"Nobody owns water", false},
                     {"quick rabbit jumps fences", false},
                     {"buy pharmaceuticals now", true},
                     {"make quick money online casino", true},
                     {"quick brown fox jumps", false}]
    
    word_pbs = NaiveBayes.train(training_data)               
    IO.inspect(word_pbs)
    assert NaiveBayes.spam_prob(word_pbs, "rabbit quick") < 0.5
    assert NaiveBayes.spam_prob(word_pbs, "quick money") > 0.5
    assert NaiveBayes.spam_prob(word_pbs, "brown rabbit") < 0.5
    assert NaiveBayes.spam_prob(word_pbs, "buy casino") > 0.5
    assert NaiveBayes.spam_prob(word_pbs, "Nobody buy brown water") < 0.5
    assert NaiveBayes.spam_prob(word_pbs, "now owns online pharmaceuticals") > 0.5
  end
end
