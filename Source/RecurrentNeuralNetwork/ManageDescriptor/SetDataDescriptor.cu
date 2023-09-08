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

  int vector_size {
    parameters.hidden_size_ * parameters.get_bidirectional_scale()};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDescriptor_v8
  // Recurrent projection can be enabled for LSTM cells and
  // CUDNN_RNN_ALGO_STANDARD only.

  if ((parameters.cell_mode_ == CUDNN_LSTM) &&
    (parameters.algo_ == CUDNN_RNN_ALGO_STANDARD) &&
    (parameters.hidden_size_ > parameters.projection_size_))
  {
    vector_size =
      parameters.projection_size_ * parameters.get_bidirectional_scale();
  }

  handle_set_descriptor(cudnnSetRNNDataDescriptor(
    descriptor.descriptor_,
    parameters.data_type_,
    layout_,
    parameters.maximum_sequence_length_,
    parameters.batch_size_,
    vector_size,
    sequence_length_array.sequence_length_array_,
    reinterpret_cast<void*>(&padding_fill_)));

  return handle_set_descriptor;
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork