#include "OutputDescriptor.h"

#include "RecurrentNeuralNetwork/ManageDescriptor/SetDataDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"

#include <stdexcept>

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

OutputDescriptor::OutputDescriptor(
  RecurrentNeuralNetwork::Parameters& parameters,
  RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array):
  y_data_descriptor_{}
{
  SetDataDescriptor set_data_descriptor {parameters.layout_};

  const auto handle_set_data_descriptor =
    set_data_descriptor.set_descriptor_for_output(
      y_data_descriptor_,
      parameters,
      sequence_length_array);

  if (!handle_set_data_descriptor.is_success())
  {
    throw std::runtime_error(handle_set_data_descriptor.get_error_message());
  }
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
