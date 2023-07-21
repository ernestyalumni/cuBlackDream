#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "Forward.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DataDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using DeepNeuralNetwork::CuDNNLibraryHandle;
using RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::Descriptor;
using RecurrentNeuralNetwork::SequenceLengthArray;
using RecurrentNeuralNetwork::WorkAndReserveSpaces;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace RecurrentNeuralNetwork
{
namespace Operations
{

HandleUnsuccessfulCuDNNCall forward(
  CuDNNLibraryHandle& handle,
  Descriptor& rnn_descriptor,
  SequenceLengthArray& sequence_length_array,
  DataDescriptor& x_data_descriptor,
  DataDescriptor& y_data_descriptor,
  WorkAndReserveSpaces& work_and_reserve_spaces)
{
  HandleUnsuccessfulCuDNNCall handle_forward {
    "Failed to run forward operation"};

  handle_forward(
    cudnnRNNForward(
      handle.handle_,
      rnn_descriptor.descriptor_));

  return handle_forward;
}

} // namespace Operations
} // namespace RecurrentNeuralNetwork
