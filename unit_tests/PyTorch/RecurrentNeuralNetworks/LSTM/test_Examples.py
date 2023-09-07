from CuISOX.PyTorch.RecurrentNeuralNetworks.LSTM.Examples import (
    LSTMWithLinearOutput)
from math import log, exp

import pytest
import torch

def test_LSTMWithLinearOutput_constructs():
  lstm_with_linear_out = LSTMWithLinearOutput(69, 42, 2, 11, 28)
  assert lstm_with_linear_out.hidden_dim == 42
  assert lstm_with_linear_out.layer_dim == 2

  assert lstm_with_linear_out.lstm.weight_ih_l0.size() == (4* 42, 69)
  assert lstm_with_linear_out.lstm.bias_ih_l0.size() == (4 * 42,)

  assert lstm_with_linear_out.fc.weight.size() == (11, 42)
  assert lstm_with_linear_out.fc.bias.size() == (11,)

def test_LSTMWithLinearOutput_has_parameters():
  lstm_with_linear_out = LSTMWithLinearOutput(42, 69, 2, 70, 29)
  parameters = lstm_with_linear_out.parameters()
  parameters_list = list(parameters)
  assert len(parameters_list) == 10
  assert parameters_list[0].size() == (4 * 69, 42)
  assert parameters_list[1].size() == (4 * 69, 69)
  assert parameters_list[2].size() == (4 * 69,)
  assert parameters_list[3].size() == (4 * 69,)
  assert parameters_list[4].size() == (4 * 69, 69)
  assert parameters_list[5].size() == (4 * 69, 69)
  assert parameters_list[6].size() == (4 * 69,)
  assert parameters_list[7].size() == (4 * 69,)
  assert parameters_list[8].size() == (70, 69)
  assert parameters_list[9].size() == (70,)

def test_LSTMWithLinearOutput_has_parameters_on_simple_classification_problem():
  lstm_with_linear_out = LSTMWithLinearOutput(4, 3, 1, 1, 27)
  parameters = lstm_with_linear_out.parameters()
  parameters_list = list(parameters)
  assert len(parameters_list) == 6
  assert parameters_list[0].size() == (4 * 3, 4)
  assert parameters_list[1].size() == (4 * 3, 3)
  assert parameters_list[2].size() == (4 * 3,)
  assert parameters_list[3].size() == (4 * 3,)
  assert parameters_list[4].size() == (1, 3)
  assert parameters_list[5].size() == (1,)

def test_LSTMWithLinearOutput_cross_entropy_loss_on_simple_classification():
  """
  @brief Test cross entropy loss on a simple classification problem.
  """
  lstm_with_linear_out = LSTMWithLinearOutput(4, 3, 1, 1, 1)

  input_1 = torch.tensor([[2.5, -1.3, 0.8]])
  labels_1 = torch.tensor([0,])

  loss_1 = lstm_with_linear_out.loss_and_optimizer.cross_entropy_loss(
    input_1, labels_1)

  assert type(loss_1) is torch.Tensor  
  assert loss_1.shape == ()
  assert loss_1.size() == ()

  denominator_1 = sum([exp(element) for element in input_1[0]])

  expected_1 = - log( exp(2.5) / denominator_1)

  assert pytest.approx(loss_1.item()) == pytest.approx(expected_1)

  input_2 = torch.tensor([[2.5, -1.3, 0.8], [1.2, 0.5, -0.7]])
  labels_2 = torch.tensor([0,1])

  loss_2 = lstm_with_linear_out.loss_and_optimizer.cross_entropy_loss(
    input_2, labels_2)

  denominator_2 = sum([exp(element) for element in input_2[1]])
  expected_2 = - log( exp(0.5) / denominator_2)

  assert loss_2.shape == ()
  assert loss_2.size() == ()

  # Reduction is mean by default, i.e. \sum_{n=1}^N l_n / N

  assert (pytest.approx(loss_2.item()) ==
    pytest.approx((expected_1 + expected_2)/2.))