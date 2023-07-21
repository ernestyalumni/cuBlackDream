#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/SetDropoutDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/SetRNNDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"

#include <stdexcept>

using RecurrentNeuralNetwork::ManageDescriptor::SetDropoutDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::set_rnn_descriptor;
using RecurrentNeuralNetwork::Parameters;

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

LibraryHandleDropoutRNN::LibraryHandleDropoutRNN(
  const Parameters parameters,
  const float dropout_probability,
  const unsigned long long seed):
  handle_{},
  dropout_descriptor_{},
  descriptor_{},
  dropout_probability_{dropout_probability},
  seed_{seed}
{
  dropout_descriptor_.get_states_size_for_forward(handle_);

  SetDropoutDescriptor set_dropout_descriptor {dropout_probability, seed};
  set_dropout_descriptor.set_descriptor(dropout_descriptor_, handle_);

  const auto result =
    set_rnn_descriptor(descriptor_, parameters, dropout_descriptor_);

  if (!result.is_success())
  {
    throw std::runtime_error(
      "Failed to set RNN descriptor in LibraryHandleDropoutRNN");
  }
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork