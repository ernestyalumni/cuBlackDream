#include "DataDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "SetDataDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;
using RecurrentNeuralNetwork::Parameters;
using RecurrentNeuralNetwork::SequenceLengthArray;

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

SetDataDescriptor::SetDataDescriptor(const cudnnRNNDataLayout_t layout):
  layout_{layout},
  padding_fill_{0.0}
{}

HandleUnsuccessfulCuDNNCall SetDataDescriptor::set_descriptor_for_input(
  DataDescriptor& descriptor,
  const Parameters& parameters,
  const SequenceLengthArray& sequence_length_array)
{
  HandleUnsuccessfulCuDNNCall handle_set_descriptor {
    "Failed to set RNN Data descriptor"};

  handle_set_descriptor(cudnnSetRNNDataDescriptor(
    descriptor.descriptor_,
    parameters.data_type_,
    layout_,
    parameters.maximum_sequence_length_,
    parameters.batch_size_,
    parameters.input_size_,
    sequence_length_array.sequence_length_array_,
    reinterpret_cast<void*>(&padding_fill_)));

  return handle_set_descriptor;
}

HandleUnsuccessfulCuDNNCall SetDataDescriptor::set_descriptor_for_output(
  DataDescriptor& descriptor,
  const Parameters& parameters,
  const SequenceLengthArray& sequence_length_array)
{
  HandleUnsuccessfulCuDNNCall handle_set_descriptor {
    "Failed to set RNN Data descriptor"};

  handle_set_descriptor(cudnnSetRNNDataDescriptor(
    descriptor.descriptor_,
    parameters.data_type_,
    layout_,
    parameters.maximum_sequence_length_,
    parameters.batch_size_,
    parameters.hidden_size_ * parameters.get_bidirectional_scale(),
    sequence_length_array.sequence_length_array_,
    reinterpret_cast<void*>(&padding_fill_)));

  return handle_set_descriptor;
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork